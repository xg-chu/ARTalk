#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import json
import torch
import argparse
import torchaudio
import numpy as np
import gradio as gr
from gtts import gTTS
from tqdm import tqdm

from artalk import BitwiseARModel
from artalk.flame_model import FLAMEModel, RenderMesh
from artalk.utils_videos import write_video

class ARTAvatarInferEngine:
    def __init__(self, load_gaga=False, fix_pose=False, clip_length=750, device='cuda'):
        self.device = device
        self.fix_pose = fix_pose
        self.clip_length = clip_length
        audio_encoder = 'wav2vec'
        ckpt = torch.load('./assets/ARTalk_{}.pt'.format(audio_encoder), map_location='cpu', weights_only=True)
        configs = json.load(open("./assets/config.json"))
        configs['AR_CONFIG']['AUDIO_ENCODER'] = audio_encoder
        self.ARTalk = BitwiseARModel(configs).eval().to(device)
        self.ARTalk.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device)
        self.mesh_renderer = RenderMesh(image_size=512, faces=self.flame_model.get_faces(), scale=1.0)
        
        self.output_dir = 'render_results/ARTAvatar_{}'.format(audio_encoder)
        os.makedirs(self.output_dir, exist_ok=True)
        self.style_motion = None

        if load_gaga:
            from artalk.GAGAvatar import GAGAvatar
            self.GAGAvatar = GAGAvatar().to(device)
            self.GAGAvatar_flame = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)

    def set_style_motion(self, style_motion):
        if isinstance(style_motion, str):
            style_motion = torch.load('assets/style_motion/{}.pt'.format(style_motion), map_location='cpu', weights_only=True)
        assert style_motion.shape == (50, 106), f'Invalid style_motion shape: {style_motion.shape}.'
        self.style_motion = style_motion[None].to(self.device)

    def inference(self, audio, clip_length=None):
        audio_batch = {'audio': audio[None].to(self.device), 'style_motion': self.style_motion}
        print('Inferring motion...')
        pred_motions = self.ARTalk.inference(audio_batch, with_gtmotion=False)[0]
        clip_length = clip_length if clip_length is not None else self.clip_length
        pred_motions = self.smooth_motion_savgol(pred_motions)[:clip_length]
        if self.fix_pose:
            pred_motions[..., 100:103] *= 0.0
        print('Done!')
        pred_motions[..., 104:] *= 0.0
        return pred_motions

    def rendering(self, audio, pred_motions, shape_id="mesh", shape_code=None, save_name='ARTAvatar.mp4'):
        print('Rendering...')
        pred_images = []
        if shape_id == "mesh":
            if shape_code is None:
                shape_code = audio.new_zeros(1, 300).to(self.device).expand(pred_motions.shape[0], -1)
            else:
                assert shape_code.dim() == 2, f'Invalid shape_code dim: {shape_code.dim()}.'
                assert shape_code.shape[0] == 1, f'Invalid shape_code shape: {shape_code.shape}.'
                shape_code = shape_code.to(self.device).expand(pred_motions.shape[0], -1)
            verts = self.ARTalk.basic_vae.get_flame_verts(self.flame_model, shape_code, pred_motions, with_global=True)
            for v in tqdm(verts):
                rgb = self.mesh_renderer(v[None])[0]
                pred_images.append(rgb.cpu()[0] / 255.0)
        else:
            # assert isinstance(shape_id, str), f'Invalid shape_id type: {type(shape_id)}'
            self.GAGAvatar.set_avatar_id(shape_id)
            for motion in tqdm(pred_motions):
                batch = self.GAGAvatar.build_forward_batch(motion[None], self.GAGAvatar_flame)
                rgb = self.GAGAvatar.forward_expression(batch)
                pred_images.append(rgb.cpu()[0])
        print('Done!')
        # save video
        print('Saving video...')
        pred_images = torch.stack(pred_images)
        dump_path = os.path.join(self.output_dir, '{}.mp4'.format(save_name))
        audio = audio[:int(pred_images.shape[0]/25.0*16000)]
        write_video(pred_images*255.0, dump_path, 25.0, audio, 16000, "aac")
        print('Done!')

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        from scipy.signal import savgol_filter
        motion_np = motion_codes.clone().detach().cpu().numpy()
        motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
        motion_np_smoothed[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=9, polyorder=3, axis=0)
        return torch.tensor(motion_np_smoothed).type_as(motion_codes)


def run_gradio_app(engine):
    def process_audio(input_type, audio_input, text_input, text_language, shape_id, style_id):
        if input_type == "Audio" and audio_input is None:
            gr.Warning("Please upload an audio file")
            return None
        if input_type == "Text" and (text_input is None or len(text_input.strip()) == 0):
            gr.Warning("Please input text content") 
            return None
        if input_type == "Text":
            gtts_lang = {"English": "en", "中文": "zh", "日本語": "ja", "Deutsch": "de", "Français": "fr", "Español": "es"}
            tts = gTTS(text=text_input, lang=gtts_lang[text_language])
            tts.save("./render_results/tts_output.wav")
            audio_input = "./render_results/tts_output.wav"
        # load audio
        audio, sr = torchaudio.load(audio_input)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        # inference
        if style_id == "default":
            engine.style_motion = None
        else:
            engine.set_style_motion(style_id)
        pred_motions = engine.inference(audio)
        # render
        save_name = f'{audio_input.split("/")[-1].split(".")[0]}_{style_id.replace(".", "_")}_{shape_id.replace(".", "_")}'
        engine.rendering(audio, pred_motions, shape_id=shape_id, save_name=save_name)
        # save pred_motions
        torch.save(pred_motions.float().cpu(), os.path.join(engine.output_dir, '{}_motions.pt'.format(save_name)))
        return os.path.join(engine.output_dir, '{}.mp4'.format(save_name)), os.path.join(engine.output_dir, '{}_motions.pt'.format(save_name))

    # create the gradio app
    if hasattr(engine, 'GAGAvatar'):
        all_gagavatar_id = list(engine.GAGAvatar.all_gagavatar_id.keys())
        all_gagavatar_id = sorted(all_gagavatar_id)
    else:
        all_gagavatar_id = []
    all_style_id = [os.path.basename(i) for i in os.listdir('assets/style_motion')]
    all_style_id = sorted([i.split('.')[0] for i in all_style_id if i.endswith('.pt')])
    with gr.Blocks(title="ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model") as demo:
        gr.Markdown("""
            <center>
            <h1>ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model</h1>
            </center>

            **ARTalk generates realistic 3D head motions from given audio, including accurate lip sync, natural facial animations, eye blinks, and head poses.**
            Please refer to our [paper](https://arxiv.org/abs/2502.20323), [project page](https://xg-chu.site/project_artalk), and [github](https://github.com/xg-chu/ARTalk) for more details about ARTalk.
            The apperance is powered by [GAGAvatar](https://xg-chu.site/project_gagavatar).
            
            Usage: Upload an audio file or input text -> Select an appearance and style -> Click generate!
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Audio & Text")
                input_type = gr.Radio(choices=["Audio", "Text"], value="Audio", label="Choose input type")
                audio_group = gr.Group()
                with audio_group:
                    audio_input = gr.Audio(type="filepath", label="Input Audio")
                text_group = gr.Group(visible=False)
                with text_group:
                    text_input = gr.Textbox(label="Input Text")
                    text_language = gr.Dropdown(choices=["English", "中文", "日本語", "Deutsch", "Français", "Español"], value="English", label="Choose the language of the input text")
            with gr.Column():
                gr.Markdown("### Avatar Control")
                appearance = gr.Dropdown(
                    choices=["mesh"] + all_gagavatar_id,
                    value="mesh", label="Choose the apperance of the speaker",
                )
                style = gr.Dropdown(
                    choices=["default"] + all_style_id,
                    value="natural_0", label="Choose the style of the speaker",
                )
            with gr.Column():
                gr.Markdown("### Generated Video")
                video_output = gr.Video(autoplay=True)
                motion_output = gr.File(label="motion sequence", file_types=[".pt"])
                
        inputs = [input_type, audio_input, text_input, text_language, appearance, style]
        btn = gr.Button("Generate")
        btn.click(fn=process_audio, inputs=inputs, outputs=[video_output, motion_output])

        if hasattr(engine, 'GAGAvatar'):
            examples = [
                ["Audio", "demo/jp1.wav", None, None, "12.jpg", "curious_0"],
                ["Audio", "demo/jp2.wav", None, None, "12.jpg", "natural_3"],
                ["Audio", "demo/eng1.wav", None, None, "12.jpg", "natural_2"],
                ["Audio", "demo/eng2.wav", None, None, "12.jpg", "happy_1"],
                ["Audio", "demo/cn1.wav", None, None, "11.jpg", "natural_1"],
                ["Audio", "demo/cn2.wav", None, None, "12.jpg", "happy_2"],
                ["Text", None, "Hello, this is a demo of ARTalk! Let's create something fun together.", "English", "12.jpg", "happy_0"],
                ["Text", None, "让我们一起创造一些有趣的东西吧。", "中文", "12.jpg", "natural_0"],
            ]
        else:
            examples = [
                ["Audio", "demo/jp1.wav", None, None, "mesh", "curious_0"],
                ["Audio", "demo/jp2.wav", None, None, "mesh", "natural_3"],
                ["Audio", "demo/eng1.wav", None, None, "mesh", "natural_2"],
                ["Audio", "demo/eng2.wav", None, None, "mesh", "happy_1"],
                ["Audio", "demo/cn1.wav", None, None, "mesh", "natural_1"],
                ["Audio", "demo/cn2.wav", None, None, "mesh", "happy_2"],
                ["Text", None, "Hello, this is a demo of ARTalk! Let's create something fun together.", "English", "mesh", "happy_0"],
                ["Text", None, "让我们一起创造一些有趣的东西吧。", "中文", "mesh", "natural_0"],
            ]
        gr.Examples(examples=examples, inputs=inputs, outputs=video_output)

        def toggle_input(choice):
            if choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        input_type.change(
            fn=toggle_input, inputs=[input_type], outputs=[audio_group, text_group]
        )

    demo.launch(server_name="0.0.0.0", server_port=8960)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', '-a', default=None, type=str)
    parser.add_argument('--clip_length', '-l', default=750, type=int)
    parser.add_argument("--shape_id", '-i', default='mesh', type=str)
    parser.add_argument("--style_id", '-s', default='default', type=str)

    parser.add_argument("--run_app", action='store_true')
    args = parser.parse_args()

    engine = ARTAvatarInferEngine(load_gaga=True, fix_pose=False, clip_length=args.clip_length)
    if args.run_app:
        run_gradio_app(engine)
    else:
        shape_id = 'mesh' if args.shape_id not in engine.GAGAvatar.all_gagavatar_id.keys() else args.shape_id
        audio, sr = torchaudio.load(args.audio_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

        base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        save_name = f'{base_name}_{args.style_id.replace(".", "_")}_{args.shape_id.replace(".", "_")}'
        engine.set_style_motion(args.style_id)
        pred_motions = engine.inference(audio)
        engine.rendering(audio, pred_motions, shape_id=args.shape_id, save_name=save_name)
