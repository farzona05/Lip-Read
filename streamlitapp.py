import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
from moviepy.editor import VideoFileClip

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('Модель машинного обучения для предсказывания текста по чтению по губам. Сначала модель переводит видео формата mpg в GiF, а потом делает по нему предсказание.')
    st.info('Farzona projects')
st.title('LipNet Full Stack App')
# Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))  # List of MPG files
selected_video = st.selectbox('Выберите видео ', options)

# Convert MPG to MP4
mpg_path = os.path.join('..', 'data', 's1', selected_video)
mp4_path = os.path.join('..', 'data', 's1', selected_video.replace('.mpg', '.mp4'))

# Convert MPG to MP4
clip = VideoFileClip(mpg_path)
clip.write_videofile(mp4_path)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('Выбранное Вами видео в формате mp4:')
        video_file = open(mp4_path, 'rb').read()
        st.video(video_file)

    with col2:
        st.info('Этот гиф всё, что видит модель машинного обучения при прогнозировании. GIF выбранного видео:')
        video, annotations = load_data(tf.convert_to_tensor(mp4_path))

        # Convert the video data to a format suitable for saving as GIF
        first_frame = video[0].numpy()

        gif_data = (first_frame.astype(np.uint8) * 255).squeeze()

        gif_file_path = f'animation_{selected_video[:-4]}.gif'
        imageio.mimsave(gif_file_path, [gif_data], duration=500)
        st.image(gif_file_path, width=400)

        st.info('Это выходные данные ML модели в виде токенов.')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Декодирование необработанных токенов в слова. Этот текст прогнозировала наша модель по GIF выбранного видео:')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

