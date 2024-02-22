import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('Модель машинного обучения для предсказывания текста по чтению по губам. Сначала модель переводит видео формата mpg в GiF, a потом делает по нему предсказание.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Выберите видео ', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('На видео ниже показано всего одно тестовое видео в формате mp4.')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('Этот гиф всё, что видит модель машинного обучения при прогнозировании. Вот один пример тестового гиф')
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # Convert the video data to a format suitable for saving as GIF
        # For example, you might take the first frame of the video
        # and convert it to a GIF
        first_frame = video[0].numpy()  # Convert EagerTensor to numpy array

        # Convert the first frame to uint8 and scale it to the range [0, 255]
        gif_data = (first_frame.astype(np.uint8) * 255).squeeze()

        # Save the GIF using imageio
        imageio.mimsave('./animation.gif', [gif_data], duration=100)  # Wrap gif_data in a list
        st.image('animation.gif', width=400) 

        st.info('Это выходные данные выбранного Вами модели машинного обучения в виде токенов.')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Декодирование необработанных токенов в слова. Этот текст прочитала наша модель по всего лишь GIF выбранного вами видео:')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

