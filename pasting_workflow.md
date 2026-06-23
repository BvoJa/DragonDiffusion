# Object Pasting Workflow — Complete Code Walkthrough

The object pasting workflow takes an object from a **reference image** and pastes it onto a **background image** at a target location, letting the diffusion model harmonize the result. Below is the full pipeline traced through every code file.

---

## 1. UI Entry Point — `src/demo/demo.py` (line 658)

```python
run_button.click(fn=runner, inputs=[...], outputs=[output])
```

When the user clicks "Run" in the Gradio UI, the `runner` function is called with all inputs: background image, mask, reference image, prompts, offsets `(dx, dy)`, `resize_scale`, weights, and diffusion parameters.

---

## 2. Orchestrator — `src/demo/model.py`, `DragonModels.run_paste()` (lines 335-413)

This is the main function that orchestrates the entire pasting pipeline. Here's each step annotated:

### Step A: Image Loading & Preprocessing (lines 336-349)

```python
def run_paste(self, img_base, mask_base, img_replace, prompt, prompt_replace,
              w_edit, w_content, seed, guidance_scale, energy_scale, 
              dx, dy, resize_scale, max_resolution, SDE_strength, ip_scale=None):
    seed_everything(seed)
    energy_scale = energy_scale * 1e3  # Scale up the energy guidance weight
    
    # Resize background image to max area (e.g. 512*768 pixels)
    img_base, input_scale = resize_numpy_image(img_base, max_resolution*max_resolution)
    h, w = img_base.shape[1], img_base.shape[0]
    img_base = Image.fromarray(img_base)
    
    # Prepare IP-Adapter image prompts (256x256 for CLIP)
    img_prompt_base = img_base.resize((256, 256))
    # Normalize from [0,1] to [-1,1] as SD expects
    img_base_tensor = (PILToTensor()(img_base) / 255.0 - 0.5) * 2
    img_base_tensor = img_base_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

    # Same preprocessing for the reference image
    img_replace = Image.fromarray(img_replace)
    img_prompt_replace = img_replace.resize((256, 256))
    img_replace = img_replace.resize((img_base_tensor.shape[-1], img_base_tensor.shape[-2]))
    img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
    img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

    mask_base = np.repeat(mask_base[:,:,None], 3, 2) if len(mask_base.shape)==2 else mask_base
```

**Purpose:** Loads both images, normalizes them into SD's [-1,1] range, and moves them to GPU. The background is resized to meet the max resolution constraint; the reference is resized to match the background's spatial dimensions. The mask is expanded from 1-channel to 3-channel if needed.

### Step B: IP-Adapter Image Embeddings (lines 353-356)

```python
emb_im_base, emb_im_uncond_base = self.editor.get_image_embeds(img_prompt_base)
emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
emb_im = torch.cat([emb_im_base, emb_im_replace], dim=1)
emb_im_uncond = torch.cat([emb_im_uncond_base, emb_im_uncond_replace], dim=1)
```

**Purpose:** Both images are passed through the IP-Adapter to create image-conditioning embeddings. The background and reference embeddings are concatenated along the sequence dimension. These will be appended to the text embeddings so the UNet can attend to both images during denoising.

### Step C: IP-Adapter Scale Update (lines 358-360)

```python
if ip_scale is not None and ip_scale != self.ip_scale:
    self.ip_scale = ip_scale
    self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
```

**Purpose:** If the user changed the IP-Adapter influence scale, reload the adapter with the new weight.

### Step D: VAE Encoding → Latent Space (lines 361-374)

```python
latent_base = self.editor.image2latent(img_base_tensor)

if resize_scale != 1:
    hr, wr = img_replace_tensor.shape[-2], img_replace_tensor.shape[-1]
    img_replace_tensor = F.interpolate(img_replace_tensor, 
                                        (int(hr*resize_scale), int(wr*resize_scale)))
    pad_size_x = abs(img_replace_tensor.shape[-1]-wr)//2
    pad_size_y = abs(img_replace_tensor.shape[-2]-hr)//2
    if resize_scale > 1:
        # Enlarge → center crop back to original size
        img_replace_tensor = img_replace_tensor[:,:,pad_size_y:pad_size_y+hr, 
                                                          pad_size_x:pad_size_x+wr]
    else:
        # Shrink → center-pad back to original size
        temp = torch.zeros(1,3,hr, wr).to(self.device, dtype=self.precision)
        temp[:,:,pad_size_y:pad_size_y+img_replace_tensor.shape[-2],
                pad_size_x:pad_size_x+img_replace_tensor.shape[-1]] = img_replace_tensor
        img_replace_tensor = temp

latent_replace = self.editor.image2latent(img_replace_tensor)
```

**Purpose:** Both images are encoded from pixels to SD's latent space via the VAE encoder (downsampling 8x, converting 3×H×W to 4×H/8×W/8). If `resize_scale != 1`, the reference image is resized and either center-cropped or center-padded so its spatial dimensions still match the background's.

### Step E: DDIM Inversion (lines 375-376)

```python
ddim_latents = self.editor.ddim_inv(
    latent=torch.cat([latent_base, latent_replace]),
    prompt=[prompt, prompt_replace]
)
latent_in = ddim_latents[-1][:1].squeeze(2)
```

**Purpose:** Both latents are **concatenated** (batch=2) and inverted together through the same DDIM inversion process. This produces a trajectory `ddim_latents[t][b]` where:
- `b=0` → background latent trajectory
- `b=1` → reference latent trajectory
- `t=0..T` → timesteps from clean to noisy

`latent_in = ddim_latents[-1][:1].squeeze(2)` takes the **final noisy background latent** (batch=1, no frame dim) as the starting point for denoising.

**DDIM Inversion** (`src/utils/inversion.py`): Takes a clean latent and runs the *reverse* of the DDIM sampling process — from clean toward pure noise — storing every intermediate step. This trajectory is used later as the memory bank for attention and as reference features for guidance.

### Step F: Mask Processing (lines 378-396)

```python
scale = 8 * SIZES[max(self.up_ft_index)] / self.up_scale
edit_kwargs = process_paste(
    path_mask=mask_base, h=h, w=w, dx=dx, dy=dy, 
    scale=scale, input_scale=input_scale, up_scale=self.up_scale,
    up_ft_index=self.up_ft_index, w_edit=w_edit, w_content=w_content,
    precision=self.precision, resize_scale=resize_scale
)
```

`process_paste()` in `src/utils/utils.py` (lines 199-240) does:

1. **Scale offsets** by `input_scale`: `dx, dy = dx*input_scale, dy*input_scale`

2. **Resize the mask** to the working image resolution `(h, w)` and threshold it:
   ```python
   mask_base = cv2.resize(mask_base, (h, w))
   mask_base = img2tensor(mask_base)[0][None, None]
   mask_base = (mask_base > 0.5).to('cuda', dtype=precision)
   ```

3. **Apply resize_scale** to the mask exactly like the reference tensor (for consistency):
   ```python
   if resize_scale is not None and resize_scale != 1:
       mask_base = F.interpolate(mask_base, (int(hi*resize_scale), int(wi*resize_scale)))
       # ... center crop or center pad ...
   ```

4. **Create two masks** — one for the source location (reference image coordinates) and one for the target location (background coordinates):
   ```python
   mask_replace = mask_base.clone()       # source: where the object is in reference
   mask_base = torch.roll(mask_base, (int(dy), int(dx)), (-2,-1))  # target: where it goes on background
   ```
   - `dict_mask['base']` → target paste region on the background (shifted by `dx, dy`)
   - `dict_mask['replace']` → source object region on the reference (unshifted)

5. **Downsample masks to feature-map resolution** (for the guidance function):
   ```python
   mask_base_cur = F.interpolate(mask_base, ...) > 0.5
   mask_replace_cur = torch.roll(mask_base_cur, (-int(dy/scale), -int(dx/scale)), (-2,-1))
   ```

6. Returns `edit_kwargs` dict containing all masks, weights, and scales.

### Step G: Latent Initialization (lines 394-396)

```python
mask_tmp = (F.interpolate(edit_kwargs['mask_base_cur'].float(), 
            (latent_in.shape[-2], latent_in.shape[-1])) > 0).float()
latent_tmp = torch.roll(ddim_latents[-1][1:].squeeze(2), 
             (int(dy/(w/latent_in.shape[-2])), int(dx/(w/latent_in.shape[-2]))), (-2,-1))
latent_in = (latent_in * (1 - mask_tmp) + latent_tmp * mask_tmp).to(dtype=latent_in.dtype)
```

**Purpose:** This **directly copies** the noisy reference-object latent into the background latent at the paste location. The mask is upsampled to latent resolution. `torch.roll` shifts the reference latent by `(dy, dx)` in latent-space coordinates. The result: `latent_in` is mostly the background latent, but the paste region is initialized from the reference object latent.

### Step H: Gradient-Guided Denoising (lines 398-409)

```python
latent_rec = self.editor.pipe.edit(
    mode='paste',
    emb_im=emb_im,
    emb_im_uncond=emb_im_uncond,
    latent=latent_in,
    prompt=prompt,
    guidance_scale=guidance_scale,
    energy_scale=energy_scale,
    latent_noise_ref=ddim_latents,
    SDE_strength=SDE_strength,
    edit_kwargs=edit_kwargs,
)
```

This calls `Sampler.edit()` which runs the DDIM denoising loop from noisy to clean, with three paste-specific mechanisms.

### Step I: Decode & Return (lines 410-413)

```python
img_rec = self.editor.decode_latents(latent_rec)[:, :, ::-1]
torch.cuda.empty_cache()
return [img_rec]
```

**Purpose:** The denoised latent is decoded back to pixel space via the VAE decoder, flipped BGR→RGB for display, and returned.

---

## 3. Denoising Engine — `src/models/Sampler.py`, `edit()` method

### 3a. Text & Image Conditioning Setup (lines 14-56)

```python
def edit(self, mode, emb_im, emb_im_uncond, latent, prompt, ...):
    # Encode text prompt
    text_input = self.tokenizer(prompt, padding="max_length", 
                                max_length=self.tokenizer.model_max_length, ...)
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
    
    # Append IP-Adapter image embeddings to text embeddings
    text_embeddings = torch.cat([text_embeddings, emb_im], dim=1)
    
    # Same for unconditional (negative prompt) embeddings
    uncond_embeddings = ...  # empty string
    uncond_embeddings = torch.cat([uncond_embeddings, emb_im_uncond], dim=1)
    
    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```

**Purpose:** Prepares the combined text + image conditioning. The IP-Adapter embeddings are appended to the text token embeddings along the sequence dimension, so the UNet's cross-attention layers attend to both text tokens and image tokens.

### 3b. Denoising Loop (lines 57-148)

```python
for i, t in enumerate(self.scheduler.timesteps):
    # Forward pass through UNet with paste mode
    noise_pred = self.unet(
        latent_model_input, t, encoder_hidden_states=text_embeddings,
        mode=mode, save_kv=False, mask=edit_kwargs['dict_mask'],
    ).sample
```

**Key detail:** `save_kv=False` tells the attention processor to **load** cached memory from inversion rather than saving new memory.

```python
    # Classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

### 3c. Gradient Guidance — `guidance_paste()` (lines 74-88 then calls function at line 438)

```python
if energy_scale > 0:
    # Detach latent for gradient computation
    latent_cur = latent.detach().requires_grad_(True)
    
    # Run guidance function
    if mode == 'paste':
        energy = self.guidance_paste(latent_cur, latent_noise_ref, t, 
                                      edit_kwargs, energy_scale)
    
    # Backpropagate to get gradient
    energy.backward()
    # Add gradient to noise prediction (steering the denoising)
    noise_pred += latent_cur.grad * energy_scale
```

**`guidance_paste()`** (lines 438-488) — the core energy function:

```python
def guidance_paste(self, latent_cur, latent_noise_ref, t, edit_kwargs, energy_scale):
    # 1. Extract reference features from the estimator UNet for:
    #    - background latent trajectory (b=0, even indices)
    #    - reference latent trajectory  (b=1, odd indices)
    #    - current editable latent
    
    ref_features = []
    for i in range(0, len(latent_noise_ref), 2):
        feat = self.estimator(latent_noise_ref[i], t, ...)['up_ft']
        ref_features.append(feat)
    
    replace_features = []
    for i in range(1, len(latent_noise_ref), 2):
        feat = self.estimator(latent_noise_ref[i], t, ...)['up_ft']
        replace_features.append(feat)
    
    cur_feat = self.estimator(latent_cur, t, ...)['up_ft']
    
    # 2. Compute background content loss (outside paste mask)
    loss_con = 0
    for ref_f in ref_features:
        diff = (cur_feat[0] - ref_f[0]) * (1 - edit_kwargs['mask_base_cur'])
        loss_con += diff.pow(2).sum()
    
    # 3. Compute object edit loss (inside paste mask)
    loss_edit = 0
    for rep_f in replace_features:
        diff = (cur_feat[0] - rep_f[0]) * edit_kwargs['mask_replace_cur']
        loss_edit += diff.pow(2).sum()
    
    # 4. Combine: preserve background, match reference object
    total_loss = (loss_con * w_content + loss_edit * w_edit) * energy_scale
    return total_loss
```

**Purpose:** The guidance computes gradients that tell the model:
- **Outside the paste mask**: preserve the original background features → "don't change the background"
- **Inside the paste mask**: match the reference object's features → "make the pasted region look like the object"

### 3d. Regional SDE Noise (lines 91-134)

```python
noise = torch.randn_like(latent_in)
# Regional SDE: add more noise inside paste region, less outside
latent_base = self.scheduler.add_noise(latent, noise, t)
mask_sde = F.interpolate(edit_kwargs['mask_base_cur'].float(), 
                          (latent.shape[-2], latent.shape[-1]))
latent_sde = latent * (1 - mask_sde) + latent_base * mask_sde
latent = latent_sde
```

**Purpose:** Adds controlled stochasticity only inside the paste region, giving the model flexibility to harmonize the object with its new environment while keeping the background deterministic and stable.

---

## 4. Masked Self-Attention — `src/unet/attention_processor.py` (lines 76-99)

```python
elif mode in ['appearance', 'paste']:
    if 35 >= iter_cur >= 0:  # Only for early-middle timesteps
        # Load cached KV memory from inversion (batch=2: [bg, ref])
        key_ref = attn.buffer_key[iter_cur]   # (2, N, C)
        value_ref = attn.buffer_value[iter_cur]
        
        # Split into foreground (reference object) and background
        key_fg = key_ref[1:]    # reference image features
        value_fg = value_ref[1:]
        key_bg = key_ref[:1]    # background image features
        value_bg = value_ref[:1]
        
        # Mask: only attend to reference-object tokens inside the paste region
        mask_fg = mask['replace']   # source object mask
        mask_fg = F.interpolate(...) > 0.5   # downsample to feature resolution
        # Mask: only attend to background tokens outside the paste region
        mask_bg = mask['base']      # target paste mask
        mask_bg = F.interpolate(...) < 0.5   # INVERT: background is NOT paste
        
        # Select only the foreground tokens that fall inside the object mask
        key_fg = key_fg[mask_fg.repeat(key_fg.shape[0],1,key_fg.shape[2])]
              .reshape(key_fg.shape[0], -1, key_fg.shape[2]).repeat(2,1,1)
        value_fg = value_fg[...same...]
        
        # Select only the background tokens that fall outside the paste mask
        key_bg = key_bg[mask_bg.repeat(key_bg.shape[0],1,key_bg.shape[2])]
              .reshape(key_bg.shape[0], -1, key_bg.shape[2]).repeat(2,1,1)
        value_bg = value_bg[...same...]
        
        # Concatenate: background memory + foreground object memory
        key = torch.cat([key_bg, key_fg], dim=1)
        value = torch.cat([value_bg, value_fg], dim=1)
```

**Purpose:** This is the **attention-level paste mechanism**. During self-attention in the UNet's up-blocks, the model's queries attend to:
- **Background memory** (from the background's inversion trajectory) for spatial positions outside the paste mask
- **Reference-object memory** (from the reference's inversion trajectory) for spatial positions inside the paste mask

This ensures the pasted object inherits the visual details (texture, color, structure) from the reference object while the background maintains its original identity.

### Memory Saving During Inversion (lines 101-106)

```python
if attn.updown == 'up' and save_kv:
    if not hasattr(attn, 'buffer_key'):
        attn.buffer_key = {}
        attn.buffer_value = {}
    attn.buffer_key[iter_cur] = key.cpu()
    attn.buffer_value[iter_cur] = value.cpu()
```

**Purpose:** During DDIM inversion (`save_kv=True`), the self-attention key/value pairs from every timestep are cached in CPU memory. During editing (`save_kv=False`), those cached values are loaded and used as the attention memory bank with mask-based filtering.

---

## 5. Supporting Functions

### `DragonPipeline.image2latent()` — `src/models/dragondiff.py` (lines 54-59)

```python
def image2latent(self, image):
    with torch.no_grad():
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215  # Scale factor for SD
    return latents
```

### `DragonPipeline.decode_latents()` — `src/models/dragondiff.py` (lines 44-52)

```python
def decode_latents(self, latents):
    latents = latents / 0.18215  # Reverse the scale
    with torch.no_grad():
        image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
    image = image.cpu().permute(0, 2, 3, 1).numpy() * 255
    return image[0].astype(np.uint8)
```

### `DragonPipeline.get_image_embeds()` — `src/models/dragondiff.py` (lines 79-89)

```python
def get_image_embeds(self, pil_image):
    # CLIP process the PIL image
    clip_image = self.clip_processor(images=pil_image, return_tensors="pt").pixel_values
    # Get vision features from CLIP
    image_embeds = self.image_encoder(clip_image.to(self.device)).pooler_output
    # Project to cross-attention dimension via Resampler
    image_embeds = self.image_proj_model(image_embeds)
    # Also create unconditional (zero) image embedding
    uncond_image_embeds = self.image_encoder(
        torch.zeros_like(clip_image).to(self.device)).pooler_output
    uncond_image_embeds = self.image_proj_model(uncond_image_embeds)
    return image_embeds, uncond_image_embeds
```

---

## Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      GRADIO UI (demo.py)                         │
│  User uploads: background, reference, mask, dx, dy, scale, etc. │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   DragonModels.run_paste()                       │
│                                                                   │
│  1. Preprocess images & mask (resize, normalize, to GPU)         │
│  2. Get IP-Adapter image embeddings (CLIP + Resampler)           │
│  3. VAE encode both images → latent_base, latent_replace         │
│  4. DDIM invert both latents together → ddim_latents trajectory  │
│     (background at b=0, reference at b=1)                        │
│  5. Start from noisy background latent: ddim_latents[-1][0]      │
│  6. process_paste(): create source & target masks at feat res    │
│  7. Initialize latent: paste shifted reference latent into       │
│     background latent at the target location                     │
│  8. Call Sampler.edit(mode='paste', ...)                         │
│  9. Decode result latent → image                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Sampler.edit(mode='paste')                     │
│                                                                   │
│  For each timestep t = T → 0:                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ 1. UNet forward (with masked self-attention memory)        │   │
│  │    - save_kv=False → load cached KV from inversion         │   │
│  │    - Mask background memory with mask['base'] (inverted)   │   │
│  │    - Mask reference memory with mask['replace']            │   │
│  │    - Attend to bg-memory outside paste, ref-memory inside  │   │
│  │                                                            │   │
│  │ 2. guidance_paste():                                       │   │
│  │    - Extract features from estimator UNet                  │   │
│  │    - loss_con: MSE outside mask (preserve background)      │   │
│  │    - loss_edit: MSE inside mask (match reference object)   │   │
│  │    - gradient = autograd.grad(loss * scale, latent)        │   │
│  │    - noise_pred += gradient                                │   │
│  │                                                            │   │
│  │ 3. DDIM step → next latent                                 │   │
│  │                                                            │   │
│  │ 4. Regional SDE: add noise inside paste mask only          │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Return final denoised latent                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│               VAE Decode → Output Image                          │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Works

The pasted object is harmonized through **three synergistic mechanisms**:

| Mechanism | Code Location | What It Does |
|-----------|--------------|--------------|
| **Masked Self-Attention Memory** | `attention_processor.py:76-99` | The UNet attends to real background features outside the paste area and real reference-object features inside it — providing pixel-level appearance cues |
| **Gradient Guidance** | `Sampler.py:guidance_paste()` | An energy function pushes the latent to preserve background features while matching reference-object features — providing semantic-level steering |
| **Regional SDE Noise** | `Sampler.py:91-134` | Extra stochasticity inside the paste mask lets the model explore variations to harmonize lighting, shadows, and perspective with the target scene |

The result: the pasted object inherits fine visual details (texture, edges, color) from the reference via attention, is semantically aligned via gradient guidance, and is naturally blended into the background through diffusion's denoising process.