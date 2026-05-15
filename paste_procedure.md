# Object Pasting Through Stable Diffusion

This note explains how the input images go through Stable Diffusion for object pasting in this codebase. The important inputs are:

- `img_base`: background image, the image that receives the object.
- `img_replace`: reference image, the image containing the object to paste.
- `mask_base`: object mask selected on the reference image.
- `dx`, `dy`, `resize_scale`: target placement and scale for the pasted object.

The main execution path is:

```text
create_demo_paste(...)
  -> DragonModels.run_paste(...)
  -> DragonPipeline.image2latent(...)
  -> DragonPipeline.ddim_inv(...)
  -> process_paste(...)
  -> Sampler.edit(mode="paste", ...)
  -> Sampler.guidance_paste(...)
  -> DragonPipeline.decode_latents(...)
```

## 1. Inputs Enter `run_paste`

The Object Pasting UI calls `run_paste` with the background image, object mask, reference image, prompts, paste offsets, resize scale, and diffusion parameters. Code: [src/demo/demo.py:657-658](src/demo/demo.py#L657-L658), [src/demo/model.py:335](src/demo/model.py#L335).

Inside `run_paste`, the background image is resized to the requested maximum resolution, converted to PIL, normalized from `[0, 1]` into Stable Diffusion's `[-1, 1]` tensor range, moved to CUDA, and given a batch dimension. Code: [src/demo/model.py:336-343](src/demo/model.py#L336-L343).

The reference image is also converted to PIL, resized to match the background tensor resolution, normalized to `[-1, 1]`, moved to CUDA, and batched. Code: [src/demo/model.py:345-349](src/demo/model.py#L345-L349).

The selected object mask is converted from one channel to three channels if needed, because `process_paste` expects an image-like mask. Code: [src/demo/model.py:351](src/demo/model.py#L351).

## 2. Images Also Become IP-Adapter Image Prompts

Before VAE encoding, both images are also passed through the IP-Adapter image-embedding path. `run_paste` gets image embeddings for the background and the reference image, then concatenates them so the sampler can condition on both images. Code: [src/demo/model.py:353-356](src/demo/model.py#L353-L356).

`DragonPipeline.get_image_embeds` does this by CLIP-processing the PIL image, running the CLIP vision encoder, projecting the image features with `image_proj_model`, and also creating unconditional image embeddings from a zero image. Code: [src/models/dragondiff.py:79-89](src/models/dragondiff.py#L79-L89).

Those image embeddings are later appended to the text embeddings inside `Sampler.edit`, so Stable Diffusion receives text conditioning plus image-prompt conditioning. Code: [src/models/Sampler.py:34-52](src/models/Sampler.py#L34-L52).

## 3. Pixels Become Stable Diffusion Latents

The background tensor is encoded by Stable Diffusion's VAE:

```python
latents = self.pipe.vae.encode(image)['latent_dist'].mean
latents = latents * 0.18215
```

Code: [src/demo/model.py:361](src/demo/model.py#L361), [src/models/dragondiff.py:54-59](src/models/dragondiff.py#L54-L59).

If the object should be resized, the reference tensor is resized before it is VAE-encoded. For `resize_scale > 1`, the larger reference tensor is center-cropped back to the background size. For `resize_scale < 1`, the smaller reference tensor is centered inside a zero tensor of the original size. Code: [src/demo/model.py:362-372](src/demo/model.py#L362-L372).

Then the resized reference tensor is also encoded by the VAE. Code: [src/demo/model.py:374](src/demo/model.py#L374), [src/models/dragondiff.py:54-59](src/models/dragondiff.py#L54-L59).

At this point there are two clean Stable Diffusion latents:

- `latent_base`: latent of the background image.
- `latent_replace`: latent of the reference image containing the object.

## 4. Both Latents Are DDIM-Inverted

The two latents are concatenated and inverted together:

```python
ddim_latents = self.editor.ddim_inv(
    latent=torch.cat([latent_base, latent_replace]),
    prompt=[prompt, prompt_replace]
)
```

Code: [src/demo/model.py:375](src/demo/model.py#L375).

`DragonPipeline.ddim_inv` adds a frame dimension and calls `DDIMInversion.invert`. Code: [src/models/dragondiff.py:61-64](src/models/dragondiff.py#L61-L64).

During inversion, the code tokenizes the two prompts, runs the text encoder, and repeatedly calls the UNet from clean latent toward noisy latent over `NUM_DDIM_STEPS`. It stores every intermediate latent in `all_latent`. Code: [src/utils/inversion.py:27-44](src/utils/inversion.py#L27-L44), [src/utils/inversion.py:47-58](src/utils/inversion.py#L47-L58), [src/utils/inversion.py:64-67](src/utils/inversion.py#L64-L67).

Because the background and reference latents are concatenated before inversion, `ddim_latents[-1][0]` corresponds to the noisy background latent and `ddim_latents[-1][1]` corresponds to the noisy reference latent. `run_paste` starts from the noisy background latent:

```python
latent_in = ddim_latents[-1][:1].squeeze(2)
```

Code: [src/demo/model.py:376](src/demo/model.py#L376).

## 5. The Object Mask Is Converted Into Paste-Space Masks

`run_paste` computes the feature-map scale and calls `process_paste` with the object mask, image size, offsets, resize scale, and guidance weights. Code: [src/demo/model.py:378-393](src/demo/model.py#L378-L393).

Inside `process_paste`, the mask follows the same geometry as the reference image:

1. Scale `dx` and `dy` by the image resize factor. Code: [src/utils/utils.py:199-200](src/utils/utils.py#L199-L200).
2. Resize the mask to the working image size and threshold it on CUDA. Code: [src/utils/utils.py:201-209](src/utils/utils.py#L201-L209).
3. If `resize_scale` is used, resize/crop/pad the mask exactly like the reference image tensor. Code: [src/utils/utils.py:210-220](src/utils/utils.py#L210-L220).
4. Clone the unshifted mask as `mask_replace`, then roll `mask_base` by `(dy, dx)` to create the target paste-region mask on the background. Code: [src/utils/utils.py:221-225](src/utils/utils.py#L221-L225).
5. Downsample the shifted paste mask to feature resolution as `mask_base_cur`, and derive `mask_replace_cur` by rolling back into reference-object coordinates. Code: [src/utils/utils.py:227-228](src/utils/utils.py#L227-L228).

`process_paste` returns `dict_mask`, `mask_base_cur`, `mask_replace_cur`, and the edit/content weights. Code: [src/utils/utils.py:230-240](src/utils/utils.py#L230-L240).

In short:

- `mask_base_cur`: where the pasted object should appear in the background.
- `mask_replace_cur`: where the source object is in the reference image.
- `dict_mask["base"]`: paste-region mask used by masked self-attention memory.
- `dict_mask["replace"]`: reference-object mask used by masked self-attention memory.

## 6. The Noisy Background Latent Is Seeded With the Reference Object

Before denoising starts, the code directly initializes the paste region in latent space.

It upsamples/downsamples `mask_base_cur` to latent resolution, rolls the noisy reference latent by the requested offset, and blends the shifted reference latent into the noisy background latent only inside the paste mask:

```python
mask_tmp = ...
latent_tmp = torch.roll(ddim_latents[-1][1:].squeeze(2), ...)
latent_in = latent_in * (1 - mask_tmp) + latent_tmp * mask_tmp
```

Code: [src/demo/model.py:394-396](src/demo/model.py#L394-L396).

This is the first concrete paste operation: the starting noisy latent is mostly the background image, but the target object region is initialized from the reference image latent.

## 7. Stable Diffusion Denoising Runs in `mode="paste"`

The seeded latent is passed to `Sampler.edit` with `mode="paste"`, the image-prompt embeddings, the background text prompt, the full DDIM inversion trajectory, SDE strength, and all paste masks. Code: [src/demo/model.py:398-409](src/demo/model.py#L398-L409).

`Sampler.edit` builds text conditioning, appends image-prompt conditioning, sets the scheduler timesteps, and extracts `dict_mask` from `edit_kwargs`. Code: [src/models/Sampler.py:14-56](src/models/Sampler.py#L14-L56).

For each denoising step, the sampler calls the UNet with the current latent, timestep, text/image conditioning, `mode="paste"`, `save_kv=False`, and the paste masks. Code: [src/models/Sampler.py:57-72](src/models/Sampler.py#L57-L72).

## 8. Masked Self-Attention Reuses Background and Reference Memory

During DDIM inversion, the UNet runs with the default `save_kv=True`; up-block self-attention keys and values are saved into attention buffers. Code: [src/utils/inversion.py:23-25](src/utils/inversion.py#L23-L25), [src/unet/attention_processor.py:101-106](src/unet/attention_processor.py#L101-L106).

During paste editing, `save_kv=False`, so the attention processor loads those saved keys/values. For `mode in ["appearance", "paste"]`, it splits the buffered memory:

- background memory: `key_ref[:1]`, `value_ref[:1]`
- reference memory: `key_ref[1:]`, `value_ref[1:]`

Then it masks background memory with `mask["base"]` and reference foreground memory with `mask["replace"]`, concatenating the valid background and object memories for self-attention. Code: [src/unet/attention_processor.py:76-99](src/unet/attention_processor.py#L76-L99).

This is the attention-level paste: outside the paste area the model attends to background-image memory, while the pasted object area can attend to reference-object memory.

## 9. Paste Guidance Pulls the Result Toward the Reference Object

When `energy_scale` is active, `Sampler.edit` calls `guidance_paste`. Code: [src/models/Sampler.py:74-88](src/models/Sampler.py#L74-L88).

`guidance_paste` extracts UNet features for:

- the inverted background latent trajectory, selected with `latent_noise_ref.squeeze(2)[::2]`;
- the inverted reference latent trajectory, selected with `latent_noise_ref.squeeze(2)[1::2]`;
- the current editable latent.

Code: [src/models/Sampler.py:438-462](src/models/Sampler.py#L438-L462).

It computes two losses:

- **Background content loss** outside the paste mask: compare current features with background features where `(1 - mask_base_cur)` is true. Code: [src/models/Sampler.py:463-470](src/models/Sampler.py#L463-L470).
- **Object edit loss** inside the paste mask: compare current pasted-region features with reference-object features selected by `mask_replace_cur`. Code: [src/models/Sampler.py:471-479](src/models/Sampler.py#L471-L479).

The losses become gradients with respect to the current latent, and the final guidance applies content gradients outside the paste mask and edit gradients inside it. Code: [src/models/Sampler.py:481-488](src/models/Sampler.py#L481-L488).

So the guidance tells Stable Diffusion: preserve the background outside the paste region, but make the paste region look like the reference object.

## 10. Regional SDE Noise Is Limited to the Paste Region

During denoising, the sampler can add regional SDE noise. For paste mode, it interpolates `mask_base_cur` to latent resolution and uses that mask to apply the stronger regional-noise branch inside the paste region while leaving the rest closer to the normal branch. Code: [src/models/Sampler.py:91-134](src/models/Sampler.py#L91-L134).

This gives the pasted area flexibility to harmonize with the target image while keeping the background stable.

## 11. The Final Latent Is Decoded Back to an Image

After all denoising steps, `Sampler.edit` returns the edited latent. Code: [src/models/Sampler.py:146-148](src/models/Sampler.py#L146-L148).

`run_paste` decodes the latent with the VAE, flips channels for display, clears CUDA cache, and returns the result. Code: [src/demo/model.py:410-413](src/demo/model.py#L410-L413).

`DragonPipeline.decode_latents` divides by Stable Diffusion's latent scale `0.18215`, calls the VAE decoder, clamps the output to `[0, 1]`, and converts the tensor into an image. Code: [src/models/dragondiff.py:44-52](src/models/dragondiff.py#L44-L52).

## Short Summary

The pasted object is not simply copied in pixel space. The code first encodes the background and reference images into Stable Diffusion latent space, DDIM-inverts both to noisy latents, copies the shifted reference-object latent into the background latent at the paste mask, and then denoises with three paste-specific controls:

1. **Masked attention memory** uses background memory outside the paste region and reference-object memory for the object. Code: [src/unet/attention_processor.py:76-99](src/unet/attention_processor.py#L76-L99).
2. **Paste guidance** preserves background features outside the mask and matches reference-object features inside the mask. Code: [src/models/Sampler.py:463-488](src/models/Sampler.py#L463-L488).
3. **Regional SDE** adds extra flexibility only inside the paste region. Code: [src/models/Sampler.py:91-134](src/models/Sampler.py#L91-L134).
