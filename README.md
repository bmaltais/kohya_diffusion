# HOWTO

This repo provide all the required config to generate stable diffusion images as found in this note: https://note.com/kohya_ss/n/n2693183a798e

Take the time to read the note, lot's of valuable information. This code is more powerfull than it may seem.

## Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/bmaltais/kohya_diffusion.git
cd kohya_diffusion
python -m venv --system-site-packages venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install diffusers[torch]==0.7.2
pip install transformers>=4.21.0
pip install ftfy
pip install opencv-python
pip install einops
pip install pytorch_lightning
pip install open-clip-torch==2.7.0
pip install tensorboard

pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
```

## Usage

### Basic use

Enter the following in a Powershell terminal:

```powershell
python gen_img_diffusers.py --ckpt "<path to model>" --outdir "<path to output directory>" --xformers --fp16 --interactive
```

### Generate images in bulk with a single prompt

Enter the following:

```powershell
python gen_img_diffusers_v1.py "<path to model>" --outdir "<path to output directory>" ` 
    --xformers --fp16 --images_per_prompt <number> --prompt "<prompt>"
```

### Options list

```powershell
usage: gen_img_diffusers.py [-h] [--v2] [--prompt PROMPT] [--from_file FROM_FILE] [--interactive] [--image_path IMAGE_PATH] [--mask_path MASK_PATH] [--strength STRENGTH]
                            [--images_per_prompt IMAGES_PER_PROMPT] [--outdir OUTDIR] [--sequential_file_name] [--n_iter N_ITER] [--H H] [--W W] [--batch_size BATCH_SIZE]
                            [--steps STEPS] [--sampler {ddim,pndm,lms,euler,euler_a,dpmsolver,dpmsolver++,k_lms,k_euler,k_euler_a}] [--scale SCALE] [--ckpt CKPT] [--vae VAE]
                            [--seed SEED] [--fp16] [--bf16] [--xformers] [--diffusers_xformers] [--opt_channels_last] [--hypernetwork_module HYPERNETWORK_MODULE]
                            [--hypernetwork_weights HYPERNETWORK_WEIGHTS] [--hypernetwork_mul HYPERNETWORK_MUL] [--clip_skip CLIP_SKIP]
                            [--max_embeddings_multiples MAX_EMBEDDINGS_MULTIPLES] [--clip_guidance_scale CLIP_GUIDANCE_SCALE] [--clip_image_guidance_scale CLIP_IMAGE_GUIDANCE_SCALE]   
                            [--guide_image_path GUIDE_IMAGE_PATH] [--highres_fix_scale HIGHRES_FIX_SCALE] [--highres_fix_steps HIGHRES_FIX_STEPS] [--highres_fix_save_1st]

options:
  -h, --help            show this help message and exit
  --v2                  load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む
  --prompt PROMPT       prompt / プロンプト
  --from_file FROM_FILE
                        if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む
  --interactive         interactive mode (generates one image) / 対話モード（生成される画像は1枚になります）
  --image_path IMAGE_PATH
                        image to inpaint or to generate from / img2imgまたはinpaintを行う元画像
  --mask_path MASK_PATH
                        mask in inpainting / inpaint時のマスク
  --strength STRENGTH   img2img strength / img2img時のstrength
  --images_per_prompt IMAGES_PER_PROMPT
                        number of images per prompt / プロンプトあたりの出力枚数
  --outdir OUTDIR       dir to write results to / 生成画像の出力先
  --sequential_file_name
                        sequential output file name / 生成画像のファイル名を連番にする
  --n_iter N_ITER       sample this often / 繰り返し回数
  --H H                 image height, in pixel space / 生成画像高さ
  --W W                 image width, in pixel space / 生成画像幅
  --batch_size BATCH_SIZE
                        batch size / バッチサイズ
  --steps STEPS         number of ddim sampling steps / サンプリングステップ数
  --sampler {ddim,pndm,lms,euler,euler_a,dpmsolver,dpmsolver++,k_lms,k_euler,k_euler_a}
                        sampler (scheduler) type / サンプラー（スケジューラ）の種類
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale
  --ckpt CKPT           path to checkpoint of model / モデルのcheckpointファイルまたはディレクトリ
  --vae VAE             path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ
  --seed SEED           seed, or seed of seeds in multiple generation / 1枚生成時のseed、または複数枚生成時の乱数seedを決めるためのseed
  --fp16                use fp16 / fp16を指定し省メモリ化する
  --bf16                use bfloat16 / bfloat16を指定し省メモリ化する
  --xformers            use xformers / xformersを使用し高速化する
  --diffusers_xformers  use xformers by diffusers (Hypernetworks doen't work) / Diffusersでxformersを使用する（Hypernetwork利用不可）
  --opt_channels_last   set channels last option to model / モデルにchannles lastを指定し最適化する
  --hypernetwork_module HYPERNETWORK_MODULE
                        Hypernetwork module to use / Hypernetworkを使う時そのモジュール名
  --hypernetwork_weights HYPERNETWORK_WEIGHTS
                        Hypernetwork weights to load / Hypernetworkの重み
  --hypernetwork_mul HYPERNETWORK_MUL
                        Hypernetwork multiplier / Hypernetworkの効果の倍率
  --clip_skip CLIP_SKIP
                        layer number from bottom to use in CLIP / CLIPの後ろからn層目の出力を使う
  --max_embeddings_multiples MAX_EMBEDDINGS_MULTIPLES
                        max embeding multiples, max token length is 75 * multiples / トークン長をデフォルトの何倍とするか 75*この値 がトークン長となる
  --clip_guidance_scale CLIP_GUIDANCE_SCALE
                        enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only) / CLIP guided SDを有効にしてこのscaleを適用する（サンプラーはDDIM、PNDM、LMSのみ）    
  --clip_image_guidance_scale CLIP_IMAGE_GUIDANCE_SCALE
                        enable CLIP guided SD by image, scale for guidance / 画像によるCLIP guided SDを有効にしてこのscaleを適用する
  --guide_image_path GUIDE_IMAGE_PATH
                        image to CLIP guidance / CLIP guided SDでガイドに使う画像
  --highres_fix_scale HIGHRES_FIX_SCALE
                        enable highres fix, reso scale for 1st stage / highres fixを有効にして最初の解像度をこのscaleにする
  --highres_fix_steps HIGHRES_FIX_STEPS
                        1st stage steps for highres fix / highres fixの最初のステージのステップ数
  --highres_fix_save_1st
                        save 1st stage images for highres fix / highres fixの最初のステージの画像を保存する
```

## Change history

* 11/25 (v3.1): Initial commit. Include support for basic SD2.0