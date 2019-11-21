'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm


def center_crop(image):
    h, w, c = np.shape(image)
    if h > w:
        dh = (h-w)//2
        image = image[dh:dh+w, :, :]
    elif w > h:
        dw = (w-h)//2
        image = image[:, dw:dw+h, :]
        
    return image


def limit_720p(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(name_list, output_folder, model_dir):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    outputs1 = network.unet_generator(input_photo)
    outputs2 = guided_filter.guided_filter(input_photo, outputs1, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    for content_path in tqdm(name_list):
        image = cv2.imread(content_path)
        
        try:
            image = limit_720p(image)
            h, w, c = np.shape(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            
            out1, out2 = sess.run([outputs1, outputs2], 
                                  feed_dict={input_photo: batch_image})

            guided_weight = 1
            output_image = (1-guided_weight)*out1+guided_weight*out2
            output_image = (np.squeeze(output_image)+1)/2
            #output_image = (output_image+0.01)**1.1+0.01
            
            output_image = np.clip(output_image*255, 0, 255).astype(np.uint8)
            
            combined = np.empty((h, w*2, 3), dtype=np.uint8)
            combined[:, :w, :] = image
            combined[:, w:, :] = output_image
            
            name = content_path.split('/')[-1].split('.')[0]
            save1_path = os.path.join(output_folder, '{}.jpg'.format(name))
            cv2.imwrite(save1_path, combined)
            #save2_path = os.path.join(output_folder, '{}_ours.jpg'.format(name))
            #cv2.imwrite(save2_path, output_image)
   
        except:
            pass
        
    

if __name__ == '__main__':
    '''
    model_dir = 'models'
    name_list = list()
    for name in os.listdir('testset'):
        name_list.append(os.path.join('testset', name))

    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder) 
    cartoonize(name_list, output_folder, model_dir)
    '''
    from selective_search import selective_search
    image = cv2.imread('testset/party7.jpg')
    output = selective_search(image, mode='single', random=False)
    print(output)


    