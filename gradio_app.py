# Multi-person Speech Transcription: pyannote_diarization_model + faster_whisper


import gradio as gr
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import subprocess
import shutil
import signal
import torch
import gc
import threading
from pyannote.core import Segment
import pandas as pd

# 创建或检查 raw_audio_wav 文件夹
raw_wav_audio_folder = 'raw_wav_audio'
output_folder = 'output/'
device = "cuda"

if not os.path.exists(raw_wav_audio_folder):
    os.makedirs(raw_wav_audio_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res:
        start = item.start
        end = item.end
        text = item.text.strip()
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = round(text_cache[0][0].start, 1)
    end = round(text_cache[-1][0].end, 1)
    return Segment(start, end), spk, sentence

def spkinfo2t(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text

def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif spk == pre_spk and text == text_cache[-1][2]:
            continue
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text

def diarize_text(transcribe_res, diarization_res):
    timestamp_text = get_text_with_timestamp(transcribe_res)
    # print(f'timestamp_text:{timestamp_text}')
    # print(f'diarization_res:{diarization_res}')
    spk_text = spkinfo2t(timestamp_text, diarization_res)
    return merge_sentence(spk_text)

def convert_or_copy_wav(input_file, stop_event):
    input_path = input_file.name
    base_name, ext = os.path.splitext(os.path.basename(input_file.name))
    output_path = os.path.join(raw_wav_audio_folder, base_name + '.wav')

    if input_file.endswith('.wav'):
        if not os.path.exists(output_path):
            print(f'Copying {input_file} to {output_path}')
            shutil.copy(input_path, output_path)
        print(f'WAV file ready...')
        return output_path
    try:
        print(f'Converting...{input_file} to {output_path}')
        process = subprocess.Popen(['ffmpeg', '-i', input_path, output_path])
        while process.poll() is None:
            if stop_event.is_set():
                process.send_signal(signal.SIGINT)
                return "转换已停止"
            stop_event.wait(1)
        return output_path
    except subprocess.CalledProcessError as e:
        return f"转换失败: {e}"

def load_model_a():
    print('loading model_a_state...')
    from pyannote.audio import Pipeline
    diarization_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    # use_auth_token="")
    diarization_model.to(torch.device(device))
    return diarization_model #, f"模型_diarization_model_加载成功:{diarization_model}"

def load_model_b():
    print('loading model_b_state...')
    from faster_whisper import WhisperModel
    transcribe_model_size = "large-v3"
    transcribe_model = WhisperModel(transcribe_model_size, device=device, compute_type="float16")
    return transcribe_model #, f"模型_transcribe_model_加载成功:{transcribe_model}"

def infer_model_a(model_a_state, audio_file_name, min_speakers, max_speakers, output):
    if model_a_state is None:
        # diarization_model
        model_a_state = load_model_a()
    print('Now model_a inferring...')
    diarization_res = model_a_state(audio_file_name, min_speakers=min_speakers, max_speakers=max_speakers)
    unique_speakers = {
        speaker for _, _, speaker in diarization_res.itertracks(yield_label=True)
    }
    detected_num_speakers = len(unique_speakers)
    # print(f'infer_b:-->diarization_res:{diarization_res}')
    output_a = '\n'+ str(diarization_res)
    return output_a, diarization_res, f'detected_num_speakers:{detected_num_speakers}', model_a_state

def infer_model_b(model_b_state, audio_file_name, diarization_res, output):
    if model_b_state is None:
        # transcribe_model
        model_b_state = load_model_b()
    transcribe_res, info = model_b_state.transcribe(audio_file_name)
    transcribe_res_list = transcribe_res
    print(f'infer_b:-->diarization_res:{diarization_res}')
    final_res = diarize_text(transcribe_res_list, diarization_res)
    output_b_item_format_list = []
    with open(f"{output_folder}{audio_file_name}.txt", "w") as txt:
        for item in final_res:
            item_format = f"{item[0]} {item[1]} {item[2]}\n"
            start_time, end_time, text =  item[0], item[1], item[2]
            output_b_item_format_list.append(item_format)
            txt.write(item_format)
    output_b_info =  f'\ntranscribe_model--info--{info}'

    # 解析数据
    parsed_data = []
    for item in final_res:
        start_time, end_time, speaker, text = item[0].start, item[0].end, item[1], item[2]
        parsed_data.append({
            'Start Time': start_time,
            'End Time': end_time,
            'Speaker': speaker,
            'Text': text
        })
    df = pd.DataFrame(parsed_data)

    return output_b_info, output_b_item_format_list, model_b_state, df

def infer_model_all(model_a_state, model_b_state, audio_file_name, min_speakers, max_speakers, output):
    output_a, diarization_res, output_a_Speaker_num, model_a_state = infer_model_a(model_a_state, audio_file_name, min_speakers, max_speakers, output)
    output_b, output_b_item_format_list, model_b_state, df = infer_model_b(model_b_state, audio_file_name, diarization_res, output)
    return output+output_a+output_b+str(output_b_item_format_list), diarization_res, output_a_Speaker_num, output_b_item_format_list, df, model_a_state, model_b_state

def unload_model_a(model_a_state, output):
    del model_a_state
    gc.collect()
    torch.cuda.empty_cache()  # 如果模型A在GPU上运行，释放GPU内存
    return gr.update(value=None), output+"\n模型A已卸载"

def unload_model_b(model_b_state, output):
    del model_b_state
    gc.collect()
    torch.cuda.empty_cache()  # 如果模型A在GPU上运行，释放GPU内存
    return gr.update(value=None), output+"\n模型B已卸载"

def check_button_available(content):
    if content:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)

def save_data(df):
    df.to_csv(output_folder+'transcript_data.csv', index=False)
    return 'Data saved successfully!'

def main():
    stop_event = threading.Event()

    def start_conversion(file, output):
        if file is not None:
            stop_event.clear()
            result = convert_or_copy_wav(file, stop_event)
        else:
            result = None
        if result is None:
            result = ""
        output += f"{result}\n"
        return result, result, output

    def stop_conversion():
        stop_event.set()
        return "转换已停止"

    def clear_cache():
        print('clear_cache...')
        # stop_conversion()
        # for filename in os.listdir(raw_wav_audio_folder):
        #     file_path = os.path.join(raw_wav_audio_folder, filename)
        #     try:
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
        #     except Exception as e:
        #         print(f'Failed to delete {file_path}. Reason: {e}')
        return gr.update(value=None), gr.update(value=None), gr.update(value=None)

    def save_data(df, output):
        df.to_csv('transcript_data.csv', index=False)
        print('Data saved successfully!')
        return output+'\nData saved successfully!'

    with gr.Blocks() as demo:
        gr.Markdown("# 音频转录")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("导入音频"):
                        gr.Markdown("## 导入音频")
                        audio_file = gr.File(label="上传音频文件", file_types=['audio'])
                    with gr.TabItem("导入视频"):
                        gr.Markdown("## 导入视频")
                        video_file = gr.File(label="上传视频文件", file_types=['video'])

                    with gr.TabItem("关于"):
                        gr.Markdown("## 关于")
                        gr.Markdown(
                            "这是一个使用 Gradio+FFmpeg+_Pyannote-Speaker-Diarization+_FastWhisper 实现的文件转换工具。\n"
                            "\n支持导入视频和音频文件，并自动转换为 WAV 格式，\n并通过Pyannote-Speaker-Diarization+_FastWhisper进行讲话人识别和文本转录。"
                        )
                with gr.Row():
                    clear_button = gr.Button("关闭当前音频并清空缓存")
                    audio_file_name = gr.Textbox(label="音频文件名")

                audio_output = gr.Audio(label="音频预览")

            with gr.Column(scale=1):
                model_a_state = gr.State()
                model_b_state = gr.State()
                # with gr.Row():
                #     load_a_btn = gr.Button("加载模型A")
                #     load_b_btn = gr.Button("加载模型B")
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Speaker_Diarization-Model-A")
                            with gr.Row():
                                min_speakers = gr.Number(label="最小说话人数")
                                max_speakers = gr.Number(label="最大说话人数")
                            infer_a_btn = gr.Button("Speaker_Diarization-推理", interactive=False)
                            output_Speaker_num = gr.Textbox(label="检测出说话人数")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Fast_Whisper-Model-B")
                            dropdown = gr.Dropdown(label="Options", choices=["large-v3", "..."])
                            infer_b_btn = gr.Button("Transcribe-推理", interactive=False)
                            output_B_info = gr.Textbox(label="Transcribe-输出-info")
                infer_all_btn = gr.Button("all-推理", interactive=False)

                diarization_res = gr.Textbox(label="Diarization结果", visible=False)
                with gr.Column():
                    unload_a_btn = gr.Button("卸载模型A")
                    unload_b_btn = gr.Button("卸载模型B")
        output = gr.Textbox(label="输出")

        gr.Markdown("# Transcript Data")
        df_b = gr.DataFrame(pd.DataFrame(columns=['Start Time', 'End Time', 'Speaker', 'Text']))

        clear_button.click(clear_cache, outputs=[audio_output, audio_file_name, audio_file])
        audio_file.change(start_conversion, inputs=[audio_file, output], outputs=[audio_file_name, output, audio_output])
        # video_file.change(start_conversion, inputs=video_file, outputs=audio_output)
        audio_file_name.change(check_button_available, inputs=audio_file_name, outputs=infer_a_btn)
        audio_file_name.change(check_button_available, inputs=audio_file_name, outputs=infer_all_btn)
        # diarization_res.change(check_button_available, inputs=diarization_res, outputs=infer_b_btn)

        infer_a_btn.click(infer_model_a, inputs=[model_a_state, audio_file_name, min_speakers, max_speakers, output],
                          outputs=[output, diarization_res, output_Speaker_num, model_a_state])
        infer_b_btn.click(infer_model_b, inputs=[model_b_state, audio_file_name, diarization_res, output],
                          outputs=[output, output_B_info, model_b_state, df_b])
        infer_all_btn.click(infer_model_all, inputs=[model_a_state, model_b_state, audio_file_name, min_speakers, max_speakers, output],
                            outputs=[output, diarization_res, output_Speaker_num, output_B_info, df_b, model_a_state, model_b_state])

        unload_a_btn.click(unload_model_a, inputs=[model_a_state, output], outputs=[model_a_state, output])
        unload_b_btn.click(unload_model_b, inputs=[model_b_state, output], outputs=[model_b_state, output])

        save_button = gr.Button("Save Data")
        save_button.click(fn=save_data, inputs=[df_b, output], outputs=output)

    demo.launch()

if __name__ == "__main__":
    main()
