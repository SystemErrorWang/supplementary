'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
import os
import numpy as np
import argparse
import network 
import loss

from tqdm import tqdm
from guided_filter import guided_filter




os.environ["CUDA_VISIBLE_DEVICES"]="0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 100000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_model_dir", default = 'train_models/model')
    parser.add_argument("--save_out_dir", default = 'train_results')
    parser.add_argument("--train_log_dir", default = 'train_trainlog')
    
    args = parser.parse_args()
    
    return args



def train(args):
    

    input_photo = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_superpixel = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    input_cartoon = tf.placeholder(tf.float32, [args.batch_size, 
                                args.patch_size, args.patch_size, 3])
    
    output = network.unet_generator(input_photo)
    output = guided_filter(input_photo, output, r=1)

    
    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)
    
    d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake, 
                                             scale=1, patch=True, name='disc_gray')
    d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake, 
                                             scale=1, patch=True, name='disc_blur')


    vgg_model = loss.Vgg19('vgg19_no_fc.npy')
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(output)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]
    
    photo_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_photo, vgg_output))/(h*w*c)
    superpixel_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_superpixel, vgg_output))/(h*w*c)
    recon_loss = photo_loss + superpixel_loss
    tv_loss = loss.total_variation_loss(output)
    
    g_loss_total = 1e4*tv_loss + 1e-1*g_loss_blur + g_loss_gray + 2e2*recon_loss
    d_loss_total = d_loss_blur + d_loss_gray

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name] 
    
    
    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('photo_loss', photo_loss)
    tf.summary.scalar('superpixel_loss', superpixel_loss)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('d_loss_gray', d_loss_gray)
    tf.summary.scalar('g_loss_gray', g_loss_gray)
    tf.summary.scalar('d_loss_blur', d_loss_blur)
    tf.summary.scalar('g_loss_blur', g_loss_blur)
    tf.summary.scalar('d_loss_total', d_loss_total)
    tf.summary.scalar('g_loss_total', g_loss_total)
      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        g_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(g_loss_total, var_list=gene_vars)
        
        d_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(d_loss_total, var_list=disc_vars)
           
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    train_writer = tf.summary.FileWriter(args.train_log_dir)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)
   
    with tf.device('/device:GPU:0'):

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('record/preunet_models'))

        face_photo_dir = 'folder_that_store_face_photos'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'folder_that_store_scenery_photos'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        face_cartoon_dir = 'folder_that_store_face_cartoon_images'
        face_cartoon_list = utils.load_image_list(face_cartoon_dir)
        scenery_cartoon_dir = 'folder_that_store_scenery_cartoon_images'
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

        for total_iter in tqdm(range(args.total_iter)):

            if np.mod(total_iter, 5) == 0: 
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)
        
            inter_out = sess.run(output, feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: photo_batch,
                                                    input_cartoon: cartoon_batch})

            '''
            adaptive coloring has to be applied with the clip_by_value 
            in the last layer of generator network, which is not very stable.
            to stabiliy reproduce our results, please use power=1.0
            and comment the clip_by_value function in the network.py first
            If this works, then try to use adaptive color with clip_by_value.
            '''
            sp_adacolor = utils.selective_adacolor(inter_out, power=1.2)
            
            _, g_loss, r_loss = sess.run([g_optim, g_loss_total, recon_loss],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: sp_adacolor,
                                                    input_cartoon: cartoon_batch})

            _, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: sp_adacolor,
                                                    input_cartoon: cartoon_batch})


            train_writer.add_summary(train_info, total_iter)
            
            if np.mod(total_iter+1, 50) == 0:

                print('folder2, adv1, iter: {}, d_loss: {}, g_loss: {}, recon_loss: {}'.\
                        format(total_iter, d_loss, g_loss, r_loss))
                if np.mod(total_iter+1, 500 ) == 0:
                    saver.save(sess, args.save_model_dir, write_meta_graph=False, global_step=total_iter)
                     
                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                    cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)

                    result_face = sess.run(output, feed_dict={input_photo: photo_face, 
                                                            input_superpixel: photo_face,
                                                            input_cartoon: cartoon_face})

                    face_superpixel = utils.selective_adacolor(result_face, power=1.2)

                    result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery, 
                                                                input_superpixel: photo_scenery,
                                                                input_cartoon: cartoon_scenery})

                    scenery_superpixel = utils.selective_adacolor(result_scenery, power=1.2)

                    utils.write_batch_image(result_face, args.save_out_dir, 
                                            str(total_iter)+'_face_result.jpg', 4)
                    utils.write_batch_image(photo_face, args.save_out_dir, 
                                            str(total_iter)+'_face_photo.jpg', 4)
                    utils.write_batch_image(face_superpixel, args.save_out_dir, 
                                            str(total_iter)+'_face_superpixel.jpg', 4)
                    utils.write_batch_image(result_scenery, args.save_out_dir, 
                                            str(total_iter)+'_scenery_result.jpg', 4)
                    utils.write_batch_image(photo_scenery, args.save_out_dir, 
                                            str(total_iter)+'_scenery_photo.jpg', 4)
                    utils.write_batch_image(scenery_superpixel, args.save_out_dir, 
                                            str(total_iter)+'_scenery_superpixel.jpg', 4)
                    

 
            
if __name__ == '__main__':
    
    args = arg_parser()
    train(args)  
   