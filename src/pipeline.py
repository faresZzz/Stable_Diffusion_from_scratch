import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# The generate function is used to generate images from the text prompt 
# and the image to image
def generate( 
        prompt,  # The text prompt
        uncond_prompt=None,  # The unconditional text prompt (used for CLIP guided diffusion also called negative text prompt)
        input_image=None,  # The input image
        strength=0.8, # The strength of the diffusion how much noise to add to the latents Inversly proportional to the strength (how much we want to pay attention to de input image)
        do_cfg=True,  # Do CLIP guided diffusion
        cfg_scale=7.5, # The scale for the CLIP guided diffusion How much to scale the difference between conditional and unconditional predictions
        sampler_name="ddpm",  # The name of the sampler (ddpm sampler)
        n_inference_steps=50,  # The number of inference steps
        models={}, 
        seed=None, 
        device=None, 
        idle_device=None, 
        tokenizer=None,
):
    # The torch.no_grad() is used to disable gradient calculation
    with torch.no_grad():

        # Check if the strength is between 0 and 1
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Lambda function to move the tensor to the idle_device
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)

        # Generate a random seed if not provided. Can also use provided seed for reproducibility
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # seletect the clip model from the models and move to the device
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            # (Batch_Size, Seq_Len) Converte cond_tokens to a tensor and move to the device
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)  Pass the cond_tokens to the clip model
            cond_context = clip(cond_tokens) 


            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids

            # (Batch_Size, Seq_Len) Converte to tensor and move to the device
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) Pass the uncond_tokens to the clip model
            uncond_context = clip(uncond_tokens)

            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim) (2 * Batch_Size, 77, 768)
            # Concatenate the conditional and unconditional context
            context = torch.cat([cond_context, uncond_context])

        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)

        # Move clip to correct device
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:

            # select the encoder model from the models and move to the device
            encoder = models["encoder"]
            encoder.to(device)

            # make the input image the same size as the WIDTH and HEIGHT  (Height, Width) -> (Height, Width) 
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))

            # Convert the image to a numpy array (Height, Width, Channel) 
            input_image_tensor = np.array(input_image_tensor)

            # Converte in to torch tensor (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

            # Rescaled the input of the image from [0:255] -> [1:1] (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # Add the batch dimension (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # Permute size to feed to the encoder (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
            # Feed the imgae to the VAE encoder to get latent space (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move to the device
            to_idle(encoder)
        else:
            # If no image provided, generate random noise (Text-to-Image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)
        
        # select the diffusion model from the models and move to the device
        diffusion = models["diffusion"]
        diffusion.to(device)


        # TQDM is used to show the progress bar
        timesteps = tqdm(sampler.timesteps)

        # Loop through the timesteps to denode the image 
        for i, timestep in enumerate(timesteps):
            # Define the timestep  (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            # If we do CLIP guided diffusion we need to repete the process twice to generate both conditional and unconditional predictions
            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by U-Net
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # We need to split the model_output into conditional and unconditional predictions
                output_cond, output_uncond = model_output.chunk(2)

                # Combine the conditional and unconditional predictions 
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove the noise using the Scheduler (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)
        
        # Move the diffusion model to the idle device
        to_idle(diffusion)

        # select the decoder model from the models and move to the device
        decoder = models["decoder"]
        decoder.to(device)

        # Decode latente to image (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        # Rescale the image from [-1:1] -> [0:255] (Batch_Size, 3, Height, Width) -> (Batch_Size, 3, Height, Width)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        # Conver to numpy array (Batch_Size, Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    # Rescale the image from old_range to new_range
    # x: (Batch_Size, Channel, Height, Width)
    # old_range: (2,)
    # new_range: (2,)
    # clamp: bool
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        # Clamp the values between new_min and new_max 
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)