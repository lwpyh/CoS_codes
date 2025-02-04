<p align="center">
    <img src="./assets/logo.jpg" width="100">
</p>



## CoS: Chain-of-Shot Prompting for Long Video Understanding
<p align="center">
    🌐 <a href="https://www.xiaohongshu.com/discovery/item/67172f5d0000000024017704?source=webshare&xhsshare=pc_web&xsec_token=GBL17lee3zbjumPCcki1x6IL0okkah9Lp3XX_IzlJwO4I=&xsec_source=pc_share" target="_blank">Website</a> | 📃 <a href="https://arxiv.org/pdf/2409.14485" target="_blank">Paper</a> | 

</p>

✨ **Highlights**:

(i) Comprehensive long video understanding. Video-XL 7B achieves the **leading performance among 7B models** on MLVU, VideoMME, VNBench and LongVideoBench.

(ii) Efficient Long visual context processing. Video-XL can process **2048 frames on an 80G GPU and achieves nearly 95% accuracy** on Needle-in-a-haystack evaluation.

(iii) Video-XL shows strong ability in some real-world scenarios, like **movie summarization, surveillance anomaly detection and Ad placement identification**.



## News
- [2024/12/22] 🔥 Most of the training data is released, including private baai-caption video data and VICO data. Feel free to use in [link](https://huggingface.co/datasets/sy1998/Video_XL_Training/tree/main). 
- [2024/10/17] 🔥 Video-XL-7B weight is released, which can process max 1024 frames. The model can process 2048 frames is around the corner.
- [2024/10/15] 🔥 Video-XL is released,  including model, training and evaluation code.

## Installation 
```bash
conda create -n CoS python=3.10 -y && conda activate cos
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
cd LongVA/
python -m pip install -e "longva/.[train]"
pip install transformers==4.46.3
pip install -q bitsandbytes==0.41.3 accelerate==0.26.0
cd lmms-eval
pip install -e .
```

## Long Video Benchmark Evaluation
For **Video-MME**, **LongVideoBench**, **MLVU** evaluation, please use  [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) After installing `lmms-eval` and CoS, you can use the following script to evaluate. note now our baseline is LongVA, you can extend our CoS to any baselines by modifying codes in lmms-eval folders.

```bash
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model longva \
    --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,video_decode_backend=decord\
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videoxl \
    --output_path ./logs/
```
<details>
<summary>Expand to see the performance on Video-MME and MLVU</summary>
<IMG src="./assets/videomme.png"/>
</details>

## Citation
If you find this repository useful, please consider giving a star :star: and citation

```
@article{shu2024video,
  title={Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding},
  author={Shu, Yan and Zhang, Peitian and Liu, Zheng and Qin, Minghao and Zhou, Junjie and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2409.14485},
  year={2024}
}
```

## Acknowledgement
- LongVA: the codebase we built upon. 
- LMMs-Eval: the codebase we used for evaluation.
- Activation Beacon: The compression methods we referring.

## License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).




