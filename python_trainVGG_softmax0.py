###############################################################################
#
#   Image Recognition with CNN 
# 
#		Copyright	:: 
#		Date        :: 2019.11.10
#		CPU         :: **
#		Language    :: Python
#		File Name   :: python_trainVGG_softmax0.py
#
###############################################################################
WORK_DIRECTORY = 'F:/work/dev/_PROGRES/python20191110'  # Work directory

import os
os.chdir( WORK_DIRECTORY ) # Change Work Directory
#os.chdir('C:/Users/rick')
#print( os.getcwd() )
#import load_images
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                # See https://qiita.com/ballforest/items/3f21bcf34cba8f048f1e
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # See https://qiita.com/tanemaki/items/e3daa1947a34f63ab6e3
#import tensorflow
#import six
import time
import psutil
import humanize
import GPUtil as GPU

GPUs = GPU.getGPUs() # XXX: only one GPU on Colab and isn't guaranteed
gpu  = GPUs[0]

def printm():
    process = psutil.Process( os.getpid() )
    print( "Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss) )
    print( "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format( gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal) )

import cv2 
#import dlib
import tensorflow as tf
import numpy as np
import keras
import gc
#import PIL
#import glob
#import re
#import argparse
import io

from   PIL                        import Image
from   scipy                      import *
from   pylab                      import *
from   shutil                     import copyfile
from   keras                      import layers
from   keras.preprocessing.image  import load_img, save_img, img_to_array, array_to_img, ImageDataGenerator
from   keras.applications         import MobileNetV2
from   keras.applications         import MobileNet
from   keras.applications         import VGG16
from   keras.models               import Sequential, Model, model_from_json, load_model, save_model
from   keras.layers               import Input, Dense, Dropout, Activation, Flatten
from   keras.layers.core          import Flatten, Dense, Dropout
from   keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from   keras.optimizers           import SGD, Adam
from   keras.utils                import np_utils, to_categorical
from   keras.metrics              import categorical_accuracy
from   keras.metrics              import binary_accuracy
from   keras.callbacks            import LearningRateScheduler
from   tensorflow.python.client   import device_lib
from   keras                      import backend as K
from   keras                      import regularizers

device_lib.list_local_devices()

config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf. Session(config=config)

###############################################################################

#------------------------------------------------------------------------------
#
# Global valiables, buffers, paths
#
#------------------------------------------------------------------------------
bach_size       =  256          # or 128
basq_size       =    8          # fixed
bach_loop       = bach_size//basq_size # 20
bach_siz4       =    4*bach_size
pair_leng       =   16
targ_size       =   96          # 96
targ_siz2       =   96          # 96
chan_size       =    3
dim_size        =  512          # 128
max_degree      =    8
max_resize      = 0.08
#------------------------------------------------------------------------------
margin_clasp    = 0.01
margin_clasn    = 1.00
margin_train    = 0.20000       # alpha margin

margin_coef0    = 0.70000       # loss range ep=0 initaial value
margin_coef1    = 0.95000       # loss range ep>0

#loss_multcoe   = 0.0025 * (bach_size-bach_loop) // bach_loop # 0.0025 from Refrence
#loss_multcop   = 0.1428 * (bach_size-bach_loop) // bach_loop # 0.0025 from Refrence
#loss_multcon   = 0.1428 * (bach_size-bach_loop) // bach_loop # 0.0025 from Refrence

loss_multcop    = 0.01500       # Anchor-Posi not used

Lrate           = 0.01000       # learning rate
Alpha0          = 0.00020       # weight decay 2e-4 not used
Alpha1          = 0.01000       # weight decay 1e-4
nb_epoch        = 2000          # epoch number
nb_steps        =  200          # steps/epoch 100,200
nb_dots         = nb_steps//10  # nb_steps/10

Scalef          = 15.0          # Scale factor
Scalef_         = 1.0/Scalef    # Scale factor(-1)

Valid0          = 5000          # VGGtest test num
Valid1          = 1024          # Softmax test num

Rocper0         =   90          # Accuracy@FAR
Rocper1         =    3          # Percentage-%
#------------------------------------------------------------------------------
margin_valid    = margin_train
ff_             = 1.0/255.0

#------------------------------------------------------------------------------
#
# Learning rate Scheduler
#
#------------------------------------------------------------------------------
def step_decay( epoch ):        # epoch:0~30,40,50
    
    r = 0.001
    if epoch >= 30: r = 0.0005
    if epoch >= 40: r = 0.0001
    
    return r
#------------------------------------------------------------------------------
#
# A sample of Evaluation Function (not used)
#
#------------------------------------------------------------------------------
def P(y_true, y_pred): 
     true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32')) 
     pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32')) 
 
     file = open( (WORK_DIRECTORY+'/mobilenet2.pred'), 'w' )
     file.write(str(y_pred))
     file.close()
 
     precision = true_positives / (pred_positives + K.epsilon()) 
     return precision
#------------------------------------------------------------------------------
#
# L2-Normalization and Scaling
#
#------------------------------------------------------------------------------
def l2norm_scaler( x ):
    return  Scalef * tf.math.l2_normalize( x, axis=1, epsilon=1e-10 )
#------------------------------------------------------------------------------
#
# Mobile Net ver.2
# Input size
# 128(128x128x3), 160（160x160x3）, 192（192x192x3）, 224（224x224x3）
#
#------------------------------------------------------------------------------
def create_mobilenet_v2( epo, wid, hei, ch, dim, num ):
    #x      = layers.Input( ( wid, hei, ch ) )
    input   = keras.layers.Input( ( wid, hei, ch ) )
    #x      = layers.UpSampling2D( 3 )( input )
    #model  = MobileNetV2( include_top=False, input_tensor=x, weights=None, pooling="avg" )
    basemdl = MobileNetV2 ( include_top      = False        ,
    #model  = VGG16       ( include_top      = False        ,
                            input_tensor     = input        ,
                            #input_shape      = ( wid, hei, ch ),
                            #alpha            = 0.5         ,
                            #depth_multiplier = 1           ,
                            #weights          = "imagenet"  ,
                            weights          = None         ,
                            pooling          = "avg"        )
    
    model   = Sequential()
    model.add( basemdl )
    model.add( Dense( dim, activation=l2norm_scaler, activity_regularizer=regularizers.l2( Alpha1 ) ) )
   #model.add( Activation( l2norm_scaler ) )
    model.add( Dense( num, activation='softmax') )
    '''
    x       = basemdl.layers[-1].output
    x       = Dense( dim, activation=l2norm_scaler )( x )
    #x       = Activation( l2norm_scaler )( x )
    x       = Dense( num, activation='softmax' )( x )
    model   = Model ( input, x )
    '''
    #--------------------------------------------------------------------------
    '''
    alpha = Alpha0  # weight decay coefficient
    for layer in model.layers:
        if   isinstance( layer, keras.layers.DepthwiseConv2D ):
            layer.add_loss( regularizers.l2(alpha)(layer.depthwise_kernel) )
        elif isinstance( layer, keras.layers.Conv2D ) or isinstance( layer, keras.layers.Dense ):
            layer.add_loss( regularizers.l2(alpha)(layer.kernel)           )
        if   hasattr   ( layer, 'bias_regularizer' ) and layer.use_bias:
            layer.add_loss( regularizers.l2(alpha)(layer.bias)             )
    '''
    #--------------------------------------------------------------------------
    model.compile(#loss         = triplet_loss              ,
                   loss         = 'categorical_crossentropy',
                  #optimizer    = 'adam'                    ,
                  #optimizer    = SGD( lr=Lrate, momentum=0.0, decay=0.00, nesterov=True  ),
                   optimizer    = SGD( lr=Lrate, momentum=0.0, decay=0.00, nesterov=False ),
                  #optimizer    = Adam(lr=Lrate, decay=0.0 ),
                  #loss_weights = [1.0,1.0]                 , # ??
                   metrics      = ['accuracy']              )
                  #metrics      = [triplet_eval]            )
    if( epo==0 ):
        model_json_str = model.to_json()
        open( 'H:/temp/mobilenet2.json','w' ).write( model_json_str )
        model.save_weights( os.path.join( 'H:/temp', 'smallcnnINI.h5' ) )
    #save_model( model, os.path.join( WORK_DIRECTORY, 'smallcnnINI.h5' ) )
    #modev = load_model( (WORK_DIRECTORY+'/smallcnn00.h5'), custom_objects={'quadlet_loss':quadlet_loss,'quadlet_eval':quadlet_eval} )
    return model
#------------------------------------------------------------------------------
#
#
#
#------------------------------------------------------------------------------
def make_clone_data                 ( img                   ,
                                      deg                   ,
                                      f_x                   ,
                                      f_y                   ,
                                      wid                   ,
                                      hei                   ):
    
    imgtmp  = cv2.resize( img, dsize=None, fx=f_x, fy=f_y )
    center  = (imgtmp.shape[1]//2, imgtmp.shape[0]//2)
    affine  = cv2.getRotationMatrix2D( center, deg, 1.0 )
    imgtmp  = cv2.warpAffine( imgtmp, affine, (imgtmp.shape[1],imgtmp.shape[0]), flags=cv2.INTER_LINEAR )
    x       = (imgtmp.shape[1]-wid)//2
    y       = (imgtmp.shape[0]-hei)//2
    imgarry = imgtmp[ y:y+hei, x:x+wid ] # Crop from x, y, w, h -> img[ y: y+h, x: x+w ]
    return imgarry
#------------------------------------------------------------------------------
#
# Set Random Data Sequence
#
#------------------------------------------------------------------------------
def set_random_data                 ( img                   , # not array
                                      flag                  , # 0:train, 1:valid
                                      loop                  ,
                                      t_v_x                 ,
                                      t_v_y                 ,
                                      a                     ):
    #--------------------------------------------------------------------------
    imgarry = img_to_array( img ) # image to array
    '''
    x       = (imgarry.shape[1]-targ_siz2)//2 # imgarry.shape[1]: 224,250
    y       = (imgarry.shape[0]-targ_siz2)//2 # imgarry.shape[0]: 224,250
    y       = y + 10 # Tuned
    imgtemp = imgarry[ y:y+targ_siz2, x:x+targ_siz2 ] # Crop from x, y, w, h -> img[ y: y+h, x: x+w ]
    t_v_x.append(imgtemp)
    '''
    t_v_x.append(imgarry)
    t_v_y.append(   a   )
    a       = a+1
    if flag==1:
        return t_v_x, t_v_y, a
    #imgchck = array_to_img( imgtemp, scale=True )
    #save_img( (WORK_DIRECTORY+'/testWWW.bmp'), imgchck )
    imgarry = imgarry[:, ::-1, :]
    t_v_x.append(imgarry) # Horizontally flip an image
    t_v_y.append(   a   ) # Horizontally flip an image
    a       = a+1
    #--------------------------------------------------------------------------
    #imgchck = array_to_img( imgtemp, scale=True )
    #save_img( (WORK_DIRECTORY+'/testXXX.bmp'), imgchck )
    for ii in range( loop ):
        
        pm      = np.random.randint( 2 )
        if( pm==0 ):
            deg =  2 + np.random.randint( max_degree )   # 8deg.
        else       :
            deg = -2 - np.random.randint( max_degree )   # 8deg.
        pm      = np.random.randint( 2 )
        if( pm==0 ):
            aff = 1.02 + max_resize * np.random.random() # 0.08
        else       :
            aff = 0.98 - max_resize * np.random.random() # 0.08
        
        imgtemp = make_clone_data( imgarry, deg, aff, aff, targ_siz2, targ_siz2 )
        t_v_x.append(imgtemp)
        t_v_y.append(   a   )
        a       = a+1
        #imgchck = array_to_img( imgtemp, scale=True )
        #save_img( (WORK_DIRECTORY+'/testYYY.bmp'), imgchck )
        imgtemp = imgtemp[:, ::-1, :]   # Horizontally flip an image
        t_v_x.append(imgtemp)           # Horizontally flip an image
        t_v_y.append(   a   )           # Horizontally flip an image
        a       = a+1
        #imgchck = array_to_img( imgtemp, scale=True )
        #save_img( (WORK_DIRECTORY+'/testZZZ.bmp'), imgchck )
    #--------------------------------------------------------------------------
    return t_v_x, t_v_y, a

###############################################################################
###############################################################################
###############################################################################
###############################################################################
#------------------------------------------------------------------------------
#
#
# M A I N  F U N C T I O N
#
#
#------------------------------------------------------------------------------
###############################################################################
###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":
    ###########################################################################
    pp      = [ 0,300,600,900,1200,1500,1800,2100,2400,2700,3000 ]
    ###########################################################################
   #qq      = [ 8,15,22,29,36,43,50,57,64,71,78,85,92, ]
   #qq      = [ 8,12,16,20,24,28,32,36,40,44,48,52,56, ]
   #rc      =  12
    printm ( )
   
    '''
    xx      = []
    yy      = []
    with open( "testVGGxxx.png", 'rb' ) as f:
        img_binary       = f.read() # binary
    img_binarystream = io.BytesIO( img_binary )
    img_pil = Image.open  ( img_binarystream  ) # img_binarystream
    imgarry = np.asarray  ( img_pil      ) # numpy uint8 (96, 96, 3)
    imgarry = np.array    ( img_pil      ) # numpy uint8 (96, 96, 3)
    imgarry = img_to_array( img_pil      ) # numpy float (96, 96, 3)
    aryan   = np.zeros( (8,targ_size,targ_size,chan_size), dtype=float )
    xx.append( imgarry )
    xx.append( imgarry )
    xx.append( imgarry )
    xx.append( imgarry )
    aryan[0]=xx[0]
    yy.append( aryan[0] )
    yy.append( imgarry )
    yy.append( imgarry )
    yy.append( imgarry )
    zz=[]
    zz=np.array(xx)
    yy      = np.array( xx[0:] )
    xx.append( yy )
    yy.append( xx )
    yy.append( imgarry )
    #yy     = np.array(  x_an[0:] ) 
    #x_an    = np.array(  x_an[0:] )                        # NG
    #x_pn    = np.array(  x_pn[0:] )                        # NG
    # np.array 化すると append,extend は NG
    '''
    RNDSeed = 0                                             # Change here!!!!
    PRENumb = '0003'                                        # Change here!!!!
    LOGNumb = '0004'                                        # Change here!!!!
    DIRPath = WORK_DIRECTORY
    #IMGPath = 'H:/work/dev/_PROGRES/_images/20190601_vggface2/'
    LOGPath = WORK_DIRECTORY + '/__log/' + LOGNumb + '/'    # Set log path
    logLoss = 'logLoss'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logLoss = os.path.join( LOGPath, logLoss )
    logLoss = logLoss.replace( "\\", "/" )
    logAccv = 'logAccv'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logAccv = os.path.join( LOGPath, logAccv )
    logAccv = logAccv.replace( "\\", "/" )
    logAcct = 'logAcct'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logAcct = os.path.join( LOGPath, logAcct )
    logAcct = logAcct.replace( "\\", "/" )
    logAccx = 'logAccx'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logAccx = os.path.join( LOGPath, logAccx )
    logAccx = logAccx.replace( "\\", "/" )
    '''
    logProg = 'logProg'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logProg = os.path.join( LOGPath, logProg )
    logProg = logProg.replace( "\\", "/" )
    '''
    logTime = 'logTime'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logTime = os.path.join( LOGPath, logTime )
    logTime = logTime.replace( "\\", "/" )
    logStat = 'logStat'+"{0:03d}".format( 0 )+'.txt'        # Set log path
    logStat = os.path.join( LOGPath, logStat )
    logStat = logStat.replace( "\\", "/" )
    open( logLoss, "w", encoding="utf-8").close             # Just create
    open( logAccv, "w", encoding="utf-8").close             # Just create
    open( logAcct, "w", encoding="utf-8").close             # Just create
    open( logAccx, "w", encoding="utf-8").close             # Just create
   #open( logProg, "w", encoding="utf-8").close             # Just create
    open( logTime, "w", encoding="utf-8").close             # Just create
    open( logStat, "w", encoding="utf-8").close             # Just create
    #--------------------------------------------------------------------------
    #
    #  Open and read 3 target files
    #
    #  F:/work/dev/_PROGRES/_images/20190520_vggface2/n000002/0001_01.jpg
    #  F:/work/dev/_PROGRES/_images/20190520_vggface2/n000002/0002_01.jpg
    #  F:/work/dev/_PROGRES/_images/20190520_vggface2/n000002/0003_01.jpg
    #
    #--------------------------------------------------------------------------
    print( "Open and read target files.................", flush=True, end='\n' )
    start   = time.time() # Processing time
    #--------------------------------------------------------------------------
    listlin = Valid0 # 5,000
    listlin_= 1.0/listlin
    fLO     = open((WORK_DIRECTORY+'/listTrainVggAll.txt')) # Change here!!!!
    fSM     = open((WORK_DIRECTORY+'/fileSumAll.txt')     ) # Change here!!!!
   #fLO     = open((WORK_DIRECTORY+'/listTrainVggDbg.txt')) # Change here!!!!
   #fSM     = open((WORK_DIRECTORY+'/fileSumDbg.txt')     ) # Change here!!!!
   #fPA     = open((WORK_DIRECTORY+'/pairlistA.txt')      )
   #fPB     = open((WORK_DIRECTORY+'/pairlistB.txt')      )
    lineLO  = fLO.readlines() # Read 1 line as string(include \n)
    lineSM  = fSM.readlines() # Read 1 line as string(include \n)
   #linePA  = fPA.readlines() # Read 1 line as string(include \n)
   #linePB  = fPB.readlines() # Read 1 line as string(include \n)
    fLO.close ()
    fSM.close ()
   #fPA.close ()
   #fPB.close ()
    #--------------------------------------------------------------------------
    # 
    #--------------------------------------------------------------------------
    vgg_bio = []
    a       = 0
    b       = 0
    for path in lineLO:
        with open( path.rstrip('\r\n'), 'rb' ) as f:
            img_binary   = f.read() # binary
        img_binarystream = io.BytesIO( img_binary )
        vgg_bio.append( img_binarystream )
        a += 1
        b += 1
        if( a==50000 ):
            a = 0
            times = time.time() - start # Processing time
            print( "... %8.2f[sec] %11d" %(times,b), flush=True, end='\n'  ) 
            start = time.time() # Processing time
    print( "Total training file: %8d.............." %b, flush=True, end='' )
    #--------------------------------------------------------------------------
    #'''
    '''
    #--------------------------------------------------------------------------
    vgg_str = []
    a       = 0
    b       = 0
    for path in lineLO:
        str_binary  = path.rstrip('\r\n')
        vgg_str.append( str_binary )
        a += 1
        b += 1
        if( a==50000 ):
            a = 0
            times = time.time() - start # Processing time
            print( "... %8.2f[sec] %11d" %(times,b), flush=True, end='\n'  ) 
            start = time.time() # Processing time
    print( "Total training file: %8d.............." %b, flush=True, end='' )
    #--------------------------------------------------------------------------
    '''
    np.random.seed( RNDSeed )                                   # random seed
    #--------------------------------------------------------------------------
    #trvgg_x = [] # append img_to_array(img)
    #trvgg_y = [] # append img_to_array(img)
    #--------------------------------------------------------------------------
    a       = 0
    for path in lineLO:
        a   = a + 1
        #path    = path.rstrip('\r\n')
        #img     = load_img( path )
        #trvgg_x, trvgg_y, a = set_random_data( img, 1, 0, trvgg_x, trvgg_y, a )
    vggnum  = a
    #--------------------------------------------------------------------------
    '''
    a       = 0
    for path in lineLO:
        #path    = os.path.join( IMGPath, path )
        #path    = path.replace( "\\", "/" ).rstrip('\r\n')
        #path    = (IMGPath + path).rstrip('\r\n')
        img = load_img( path.rstrip('\r\n') )
        trvgg_x, trvgg_y, a = set_random_data( img, 1, 0, trvgg_x, trvgg_y, a )
        if( a==listlin+100 ): break
    '''
    #--------------------------------------------------------------------------
    a       = 0
    for path in lineSM:
        a   = a + 1
    dirnum  = a
    #--------------------------------------------------------------------------
    randv   = [np.random.randint(0,dirnum) for ii in range( listlin )] # 5,000
    vaanc   = []
    vapos   = []
    vaneg   = []
    for ii in range( listlin ): # 5,000
        ar  = randv[ii] # Anchor/Positive ID
        r1  = int(lineSM[ar])
        if( ar>0 ): r0 = int(lineSM[ar-1])
        else      : r0 = 0
        Anc     = np.random.randint ( r0, r1 ) # File #
        while( 1 ):
            Pos = np.random.randint ( r0, r1 ) # File #
            if( Pos!=Anc ): break
       #img_pil = load_img    ( vgg_str[Anc] )
        img_pil = Image.open  ( vgg_bio[Anc] ) # open image
        imgarry = img_to_array( img_pil      )
        vaanc.append( imgarry )
       #img_pil = load_img    ( vgg_str[Pos] )
        img_pil = Image.open  ( vgg_bio[Pos] ) # open image
        imgarry = img_to_array( img_pil      )
        vapos.append( imgarry )
        
        while( 1 ): # Negative
            pr  = np.random.randint ( dirnum ) # Negative ID
            r1  = int(lineSM[pr])
            if( pr!=ar ):
                if( pr>0 ): r0 = int(lineSM[pr-1])
                else      : r0 = 0
                Neg     = np.random.randint( r0, r1 ) # File #
                break
       #img_pil = load_img    ( vgg_str[Neg] )
        img_pil = Image.open  ( vgg_bio[Neg] ) # open image
        imgarry = img_to_array( img_pil      )
        vaneg.append( imgarry )
        
    vavgg_x = [] # append img_to_array(img)
    for L in range( listlin ):
        vavgg_x.append( img_to_array( vaanc[L] ) )
    for L in range( listlin ):
        vavgg_x.append( img_to_array( vapos[L] ) )
    for L in range( listlin ):
        vavgg_x.append( img_to_array( vaneg[L] ) )

    vavgg_x = np.array( vavgg_x[0:] )                           # Important!!
    vavgg_x = vavgg_x.astype('float32')                         # Cast to float32
    vavgg_x*= ff_                                               # Normalize[0~1]
    
    del vaanc
    del vapos
    del vaneg
    gc.collect  ()
    #--------------------------------------------------------------------------
   #dirnum  = 1024
    #--------------------------------------------------------------------------
   #fVA     = open( (WORK_DIRECTORY+'/list2LFW_A.txt') )
   #fVB     = open( (WORK_DIRECTORY+'/list2LFW_B.txt') )
   #fVC     = open( (WORK_DIRECTORY+'/list2LFW_C.txt') )
   #fVD     = open( (WORK_DIRECTORY+'/list2LFW_D.txt') )
    fVA     = open( (WORK_DIRECTORY+'/list4LFW_A.txt') ) # Annotation New LFW
    fVB     = open( (WORK_DIRECTORY+'/list4LFW_B.txt') ) # Annotation New LFW
    fVC     = open( (WORK_DIRECTORY+'/list4LFW_C.txt') ) # Annotation New LFW
    fVD     = open( (WORK_DIRECTORY+'/list4LFW_D.txt') ) # Annotation New LFW
   #fVA     = open( (WORK_DIRECTORY+'/list8LFW_A.txt') ) # Annotation New LFW
   #fVB     = open( (WORK_DIRECTORY+'/list8LFW_B.txt') ) # Annotation New LFW
   #fVC     = open( (WORK_DIRECTORY+'/list8LFW_C.txt') ) # Annotation New LFW
   #fVD     = open( (WORK_DIRECTORY+'/list8LFW_D.txt') ) # Annotation New LFW
    lineA   = fVA.readlines() # Read 1 line as string(include \n)
    lineB   = fVB.readlines() # Read 1 line as string(include \n)
    lineC   = fVC.readlines() # Read 1 line as string(include \n)
    lineD   = fVD.readlines() # Read 1 line as string(include \n)
    fVA.close()
    fVB.close()
    fVC.close()
    fVD.close()
    valfw_x = [] # append img_to_array(img)
    valfw_y = [] # append img_to_array(img)
    a = b = 0
    for path in lineA:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
            img     = load_img   ( path )
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    a = b = 0
    for path in lineB:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
            img     = load_img   ( path )
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    a = b = 0
    for path in lineC:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
            img     = load_img   ( path )
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    a = b = 0
    for path in lineD:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
            img     = load_img   ( path )
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    lfwnum  = a
    lfwnum_ = 1.0/a
    valfw_x = np.array( valfw_x[0:] )                           # Important!!
    valfw_y = np.array( valfw_y[0:] )                           # Important!!
    valfw_x = valfw_x.astype('float32')                         # Cast to float32
    valfw_y = valfw_y.astype('float32')                         # Cast to float32
    valfw_x*= ff_                                               # [0~1] Normalize
    #--------------------------------------------------------------------------
    times = time.time() - start # Processing time
    print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )           # 283.11[sec]
    print( "Number of total training   images [%8d]" %vggnum, flush=True )  # 136038
    print( "Number of total validation images [%8d]" %lfwnum, flush=True )  # 300,600,3000
    
    #--------------------------------------------------------------------------
    #
    #  Set parameters
    #
    #--------------------------------------------------------------------------
    epoch   = nb_epoch      # epoch#
    steps   = nb_steps      # steps/epoch
    tryme   = steps//2      # tryme not used
    trmin   = 1             # training min val(>=2) not used
    human   = 0             # RESET
    hupre   = 0             # RESET
    thrsh   = 0.00          # RESET
    thrLN   = 2.00          # RESET
    thrRP   = 0.00          # RESET
    precV   = 0.00          # RESET
    precT   = 0.00          # RESET
    precX   = 0.00          # RESET
    vggn_   = 1.0/vggnum    # 1/vggnum
    #--------------------------------------------------------------------------
    #
    #  Create CN network model
    #
    #--------------------------------------------------------------------------
    
    '''
    print( "Creating CN network model..................", flush=True, end='' )
    start = time.time() # Processing time
    model = create_mobilenet_v2( ep, targ_size, targ_size, chan_size, dim_size, dirnum )
    model._make_predict_function()
    times = time.time() - start # Processing time
    print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )
    '''
    
    print( "=================================================================", flush=True, end='\n' )
    epset = time.time() # Processing time
    
    for ep in range( epoch ):

        #----------------------------------------------------------------------
        #
        #  Create CN network model
        #
        #----------------------------------------------------------------------
        print( "Creating CN network model..................", flush=True, end='' )
        start = time.time() # Processing time
        model = create_mobilenet_v2( ep, targ_size, targ_size, chan_size, dim_size, dirnum )
        model._make_predict_function()
        times = time.time() - start # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )
    
        #----------------------------------------------------------------------
        #
        #  Get Weight data from h5 file
        #
        #----------------------------------------------------------------------
        print( "Get Weight data from h5 file...............", flush=True, end='' )
        start       = time.time()                   # Processing time
        #----------------------------------------------------------------------
        if( ep==0 ):
            weifile     = 'smallcnnINI.h5'          # Log file
            #weifile    = 'smallcnn106.h5'          # Log file
            #weifile    = 'smallcnn005.h5'          # Log file
            #weifile    = 'smallcnn100.h5'          # Log file
            #weifile    = 'smallcnn200.h5'          # Log file
            #weifile    = 'smallcnn008.h5'          # Log file
            #WEIGHT_PATH = os.path.join( ('H:/temp/'+PRENumb), weifile ).replace( "\\", "/" )
            WEIGHT_PATH = os.path.join( ('H:/temp/'        ), weifile ).replace( "\\", "/" )
        else       :
            weifile     = 'smallcnn'+"{0:03d}".format( (ep-1) )+'.h5'   # Log file
            WEIGHT_PATH = os.path.join( ('H:/temp/'+LOGNumb), weifile ).replace( "\\", "/" )
        #----------------------------------------------------------------------
        if  ( ep<  20 ):
            L_rate=0.01000
        elif( ep<  50 ):
            L_rate=0.00500
        elif( ep< 100 ):
            L_rate=0.00100
        elif( ep< 200 ):
            L_rate=0.00050
        elif( ep< 400 ):
            L_rate=0.00010
        elif( ep< 600 ):
            L_rate=0.00005
        else           :
            L_rate=0.00001
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        #  Set parameters
        #
        #  xxman
        #  xxm   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,
        #  qq[]  8,12,16,20,24,28,32,36,40,44,48,52,56,
        #----------------------------------------------------------------------
        batch           = bach_size                             # RESET
        TRY             = 0                                     # RESET
        lossmea         = 0.0                                   # RESET
        verb            = 0                                     # RESET
        #----------------------------------------------------------------------
        '''
        modelname_text  = open( "H:/temp/mobilenet2.json" ).read()
        model           = model_from_json( modelname_text ) # model structure 
        #model.load_weights( os.path.join( "H:/temp", weifile ).replace( "\\", "/" ) ) # training result
        #----------------------------------------------------------------------
        alpha = Alpha  # weight decay coefficient
        for layer in model.layers:
            if   isinstance( layer, keras.layers.DepthwiseConv2D ):
                layer.add_loss( regularizers.l2(alpha)(layer.depthwise_kernel) )
            elif isinstance( layer, keras.layers.Conv2D ) or isinstance( layer, keras.layers.Dense ):
                layer.add_loss( regularizers.l2(alpha)(layer.kernel)           )
            if   hasattr   ( layer, 'bias_regularizer' ) and layer.use_bias:
                layer.add_loss( regularizers.l2(alpha)(layer.bias)             )
        #----------------------------------------------------------------------
        model.compile     ( loss      = 'categorical_crossentropy',
                            optimizer = SGD( lr=L_rate, momentum=0.0, decay=0.00, nesterov=False ),
                           #optimizer = SGD( lr=L_rate, momentum=0.0, decay=0.00, nesterov=True  ),
                           #optimizer = Adam(lr=L_rate, decay=0.0),
                            metrics   = ['accuracy']              )
        '''
        #----------------------------------------------------------------------
        model.load_weights( WEIGHT_PATH ) # training result
        #----------------------------------------------------------------------
        times = time.time() - start                             # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )
        #if( ep>0 ): K.set_learning_phase( 1 )                  # 0:Test, 1:Training
        #----------------------------------------------------------------------
        print( "[%6d] %10d: "  %( ep, human ), flush=True, end='\n' )
        print( "Now training ( batch_size:%4d )." %(batch), flush=True, end='' )
        start = time.time()                                     # Processing time
        for se in range( 1000 ):                                # while( 1 )
            #print( "[%6d][%6d] %10d:" %( ep, se, human ), flush=True, end='\n' )
            #print( "Read image data and set training data......", flush=True, end='' )
            #start      = time.time()                           # Processing time
            x_an        = []                                    # 50 same anchor append img_to_array(img)
            randv       = [np.random.randint(0,dirnum) for ii in range(batch)]
            for ii in range( batch ):                           # batch: 256
                ar      = randv[ii]                             # Anchor/Positive ID
                r1      = int(lineSM[ar])                       # Get sum val.
                if( ar>0 ): r0 = int(lineSM[ar-1])              # File# range r0~r1
                else      : r0 = 0                              # File# range r0~r1
                Anc     = np.random.randint( r0, r1 )           # File# random
                img_pil = Image.open  ( vgg_bio[Anc] )          # open images
                imgarry = img_to_array( img_pil      )          # numpy float
                x_an.append   ( imgarry )                       # append
            x_train     = np.array( x_an[0:] )                  # Important!!
            x_train     = x_train.astype('float32')             # Cast to float32
            x_train    *= ff_                                   # Normalize(0.0~1.0)
            y_train     = to_categorical( randv, dirnum )       # Convert categorical val.
            
            #times      = time.time() - start                   # Processing time
            #print( "done --- %8.2f[sec]" %(times), flush=True, end='' )
            #print( " [vgg-]:%6.2f[%%]   [lfw-]:%6.2f[%%]" %( 100.0*precT, 100.0*precV ), flush=True, end='\n' )
            #------------------------------------------------------------------
            #
            #  Train with fit() function
            #
            #------------------------------------------------------------------
            #print( "Now training ( batch_size:%4d )..........." %(batch), flush=True, end='' )
            #start      = time.time()                           # Processing time
            history     = model.fit( x_train              ,
                                     y_train              ,
                                     callbacks=None       ,
                                     shuffle=False        ,
                                     validation_split=0.0 ,
                                     validation_data=None ,
                                    #batch_size=bach_size ,
                                     steps_per_epoch=1    ,     #1000//batch_size, # invalid parameter
                                     verbose=verb         ,
                                     epochs=1             )
            
            #times       = time.time() - start                  # Processing time
            #print( "done --- %8.2f[sec]" %(times), flush=True, end='' )
            loss_hist   = history.history["loss"]
            lossmea     = lossmea + loss_hist[0]
            #print( " [loss]:%11.4f [accu]:%6.2f[%%]" %( loss_hist[0], 100.0*precX ), flush=True, end='\n' )
            #with open( logLoss, "a" ) as f_handle:
            #    f_handle.write( "%d\t%.2f\n" %( ep*steps+se, loss_hist[0] ) )
            del(  x_an  )
            gc.collect ()
            human       = human + batch                         # Increment human#
            TRY         = TRY + 1
            if( ((se+1)%nb_dots)==0 ):                          # nb_dots:depends on steps
                print( ".", flush=True, end='' )
            if( TRY>=steps ):
                lossmea = lossmea / steps                       # loss mean
                times   = time.time() - start                   # Processing time
                print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )
                break
        #----------------------------------------------------------------------
        
        #======================================================================
        #
        #  Predict validation imagesm and save logfiles
        #
        #======================================================================
        print( "PREDICT VALIDATION INAGES..................", flush=True, end='' )
        start       = time.time()                               # Processing time
        
        K.set_learning_phase( 0 )                               # 0:Test, 1:Training
        
        model_pred  = Model( inputs=model.input, outputs=model.get_layer( name=None, index=1 ).output )
        max_0       = 300
        max_1       = max_0 - 1
        max_2       = 250
        Reso        = 200.0 # Resolution
        Reso_       = 0.005 # Resolution
        #----------------------------------------------------------------------
        # LFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFW
        #----------------------------------------------------------------------
        pred    = model_pred.predict ( valfw_x, batch_size=batch, verbose=0, steps=None )
        # L2 Normalization already done in the predition process
        pred    = pred * Scalef_
        posiA   = pred[0                   :lfwnum,                     :]
        posiB   = pred[lfwnum              :lfwnum+lfwnum,              :]
        negaC   = pred[lfwnum+lfwnum       :lfwnum+lfwnum+lfwnum,       :]
        negaD   = pred[lfwnum+lfwnum+lfwnum:lfwnum+lfwnum+lfwnum+lfwnum,:]
        #----------------------------------------------------------------------
        '''
        posidis = K.sum ( K.square(posiB-posiA), axis=1, keepdims=True )
        negadis = K.sum ( K.square(negaD-negaC), axis=1, keepdims=True )
        posidis = K.sqrt( posidis )
        negadis = K.sqrt( negadis )
        '''
        posidis = K.sum( (posiA*posiB), axis=1, keepdims=True )     # Similarity
        negadis = K.sum( (negaC*negaD), axis=1, keepdims=True )     # Similarity
        disposi = 'dis_posv'+"{0:03d}".format( ep )+'.txt'          # Log file
        disnega = 'dis_negv'+"{0:03d}".format( ep )+'.txt'          # Log file
        disposi = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disposi )
        disnega = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disnega )
        histpos = [0]*max_0
        histneg = [0]*max_0
        parray  = K.eval( posidis )
        narray  = K.eval( negadis )
        pmean   = 0.0
        pvari   = 0.0
        nmean   = 0.0
        nvari   = 0.0
        for ii in  range( lfwnum ):    
            kk   = parray[ii][0] * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histpos[jj]     = histpos[jj] + 1
            pmean = pmean + parray[ii][0]
            pvari = pvari +(parray[ii][0]*parray[ii][0])
            kk   = narray[ii][0] * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histneg[jj]     = histneg[jj] + 1
            nmean = nmean + narray[ii][0]
            nvari = nvari +(narray[ii][0]*narray[ii][0])
        pmean   = pmean*lfwnum_
        pvari   = pvari*lfwnum_
        pvari   = pvari-pmean*pmean
        nmean   = nmean*lfwnum_
        nvari   = nvari*lfwnum_
        nvari   = nvari-nmean*nmean
        #----------------------------------------------------------------------
        with open( disposi, "w" ) as f_handle:
            for ii in range( max_2 ):  f_handle.write( "%lf\t%.6f\n" %( Reso_*ii, histpos[ii] ) )
        with open( disnega, "w" ) as f_handle:
            for ii in range( max_2 ):  f_handle.write( "%lf\t%.6f\n" %( Reso_*ii, histneg[ii] ) )
        with open( logStat, "a" ) as f_handle:
            f_handle.write( "%d\t%lf\t(pm,pv),(nm,nv),(PM,PV),(NM,NV) = ( %.3f, %.3f ),( %.3f, %.3f ) " %( ep, lossmea, pmean, pvari, nmean, nvari ) )
        #----------------------------------------------------------------------
        # Calculate accuraccy
        #----------------------------------------------------------------------
        '''
        jj      = 0
        kk      = lfwnum/10
        for ii in  range( max_0 ):
            jj  = jj + histneg[ii]
            if jj >= kk:
                kk = ii
                break
        if kk<max_0: kk = kk+1
        jj = 0
        for ii in  range( kk ):
            jj  = jj + histpos[ii]
        precV   = jj / lfwnum
        '''
        jj      = 0
        kk      = (Rocper0*lfwnum+50)//100
        for ii in  range( max_0 ):
            jj  = jj + histneg[ii]
            if jj >= kk:
                kk = ii
                break
       #if kk<max_0: kk = kk+1
        jj = 0
        for ii in  range( kk ):
            jj  = jj + histpos[ii]
        precV   = (lfwnum-jj) / lfwnum
        #----------------------------------------------------------------------
        # Calculate threshold
        #----------------------------------------------------------------------
        '''
        MIN     = lfwnum
        THR     = 0
        jj      = 0
        kk      = 0
        for ii in  range( max_0 ):
            jj  = jj + histpos[ii]
            kk  = kk + histneg[ii]
            sp  = lfwnum  - jj
            dif = abs( kk - sp )
            if( dif < MIN ):
                MIN = dif
                THR = ii
        #thrsh  = THR / 100.0 # UPDATE
        #----------------------------------------------------------------------
        # Calculate accuraccy
        #----------------------------------------------------------------------
        jj      = 0
        kk      = THR
        if kk<max_0: kk = kk+1
        for ii in  range( kk ):
            jj  = jj + histpos[ii]
            #if jj >= kk:
            #    kk = ii
            #    break
        precV   = jj / lfwnum
        '''
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # VGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGG
        #----------------------------------------------------------------------
        pred    = model_pred.predict ( vavgg_x, batch_size=batch, verbose=0, steps=None )
        # L2 Normalization already done in the predition process
        pred    = pred * Scalef_
        posiA   = pred[0      :listlin        ,:]
        posiB   = pred[listlin:listlin+listlin,:]
        #----------------------------------------------------------------------
        '''
        posidis = K.sum ( K.square(posiB-posiA), axis=1, keepdims=True )
        posidis = K.sqrt( posidis )
        '''
        posidis = K.sum( (posiA*posiB), axis=1, keepdims=True ) # Similarity
        disposi = 'disTposv'+"{0:03d}".format( ep )+'.txt'      # Log file
        disposi = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disposi )
        histpos = [0]*max_0
        parray  = K.eval( posidis )
        Pmean   = 0.0
        Pvari   = 0.0
        for ii in  range( listlin ):
            kk   = parray[ii][0] * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histpos[jj]     = histpos[jj] + 1
            Pmean = Pmean + parray[ii][0]
            Pvari = Pvari +(parray[ii][0]*parray[ii][0])
        Pmean   = Pmean*listlin_
        Pvari   = Pvari*listlin_
        Pvari   = Pvari-Pmean*Pmean
        #----------------------------------------------------------------------
        negaC   = pred[listlin+listlin:listlin+listlin+listlin,:]
        '''
        negadis = K.sum ( K.square(negaC-posiA), axis=1, keepdims=True )
        negadis = K.sqrt( negadis )
        '''
        negadis = K.sum( (posiA*negaC), axis=1, keepdims=True ) # Similarity
        disnega = 'disTnegv'+"{0:03d}".format( ep )+'.txt'      # Log file
        disnega = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disnega )    
        histneg = [0]*max_0
        narray  = K.eval( negadis )
        Nmean   = 0.0
        Nvari   = 0.0
        for ii in  range( listlin ):
            kk   = narray[ii][0] * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histneg[jj]     = histneg[jj] + 1
            Nmean = Nmean + narray[ii][0]
            Nvari = Nvari +(narray[ii][0]*narray[ii][0])
        Nmean   = Nmean*listlin_
        Nvari   = Nvari*listlin_
        Nvari   = Nvari-Nmean*Nmean
        #----------------------------------------------------------------------
        with open( disposi, "w" ) as   f_handle:
            for ii in range( max_2 ):  f_handle.write( "%lf\t%.6f\n" %( Reso_*ii, histpos[ii] ) )
        with open( disnega, "w" ) as   f_handle:
            for ii in range( max_2 ):  f_handle.write( "%lf\t%.6f\n" %( Reso_*ii, histneg[ii] ) )
        with open( logStat, "a" ) as   f_handle:
            f_handle.write( "( %.3f, %.3f ),( %.3f, %.3f )\n" %( Pmean, Pvari, Nmean, Nvari ) )
        #----------------------------------------------------------------------
        # Calculate accuraccy
        #----------------------------------------------------------------------
        '''
        jj      = 0
        kk      = (listlin*Rocper0)//100
        for ii in  range( max_0 ):
            jj  = jj + histneg[ii]
            if jj >= kk:
                kk = ii
                break
        if kk<max_0: kk = kk+1
        jj = 0
        for ii in  range( kk ):
            jj  = jj + histpos[ii]
        precT   = jj / listlin
        '''
        jj      = 0
        kk      = (Rocper0*listlin+50)//100
        for ii in  range( max_0 ):
            jj  = jj + histneg[ii]
            if jj >= kk:
                kk = ii
                break
       #if kk<max_0: kk = kk+1
        jj = 0
        for ii in  range( kk ):
            jj  = jj + histpos[ii]
        precT   = (listlin-jj) / listlin
        #----------------------------------------------------------------------
        # Calculate threshold
        #----------------------------------------------------------------------
        '''
        MIN     = listlin
        THR     = 0
        jj      = 0
        kk      = 0
        for ii in  range( max_0 ):
            jj  = jj + histpos[ii]
            kk  = kk + histneg[ii]
            sp  = listlin - jj
            dif = abs( kk - sp )
            if( dif < MIN ):
                MIN = dif
                THR = ii
        thrsh   = THR / 100.0 # UPDATE
        #----------------------------------------------------------------------
        # Calculate accuraccy
        #----------------------------------------------------------------------
        jj      = 0
        kk      = THR
        if kk<max_0: kk = kk+1
        for ii in  range( kk ):
            jj  = jj + histpos[ii]
            #if jj >= kk:
            #    kk = ii
            #    break
        precT   = jj / listlin
        '''
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # VGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftmax!!!!!!!!!
        #----------------------------------------------------------------------
        x_an        = []                                        # 50 same anchor append img_to_array(img)
        randv       = [np.random.randint(0,dirnum) for ii in range(Valid1)]
        for ii in range( Valid1 ):                              # Valid1: 1,024
            ar      = randv[ii]                                 # Anchor/Positive ID
            r1      = int(lineSM[ar])                           # Get sum val.
            if( ar>0 ): r0 = int(lineSM[ar-1])                  # File# range r0~r1
            else      : r0 = 0                                  # File# range r0~r1
            Anc     = np.random.randint( r0, r1 )               # File# random
            img_pil = Image.open  ( vgg_bio[Anc] )              # open images
            imgarry = img_to_array( img_pil      )              # numpy float
            x_an.append   ( imgarry )                           # append
        x_valid     = np.array( x_an[0:] )                      # Important!!
        x_valid     = x_valid.astype('float32')                 # Cast to float32
        x_valid    *= ff_                                       # Normalize(0.0~1.0)
        y_valid     = to_categorical( randv, dirnum )           # Convert categorical val.
        pred        = model.predict ( x_valid, batch_size=batch, verbose=0, steps=None )
        jj          = 0
        for ii in range( Valid1 ):                              # Valid1: 1,024
            if( pred[ii].argmax()==y_valid[ii].argmax() ): jj  = jj+1
        precX       = jj/Valid1                                 # Accuracy
        times       = time.time() - start                       # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='' )
        print( " [LOSS]:%11.4f [ACCU]:%6.2f[%%]" %( lossmea, 100.0*precX ), flush=True, end='\n' )
        #----------------------------------------------------------------------
        
        del( x_an    )
        del( x_valid )
        del( y_valid )
        del( pred    )
        del( posiA   )
        del( posiB   )
        del( negaC   )
        del( negaD   )
        del( posidis )
        del( negadis )
        del( histpos )
        del( histneg )
        del( parray  )
        del( narray  )
        gc.collect  ()
        
        #----------------------------------------------------------------------
        EPOCH = human * vggn_
        with open( logAccv, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, precV   ) )
        with open( logAcct, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, precT   ) )
        with open( logAccx, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, precX   ) )
        with open( logLoss, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, lossmea ) )
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        #  Save parameters to data file smallcnnXXX.h5
        #
        #----------------------------------------------------------------------
        print( "Save parameters to data file...............", flush=True, end='' )
        start   = time.time() # Processing time
        #weifile = 'smallcnn'+"{0:03d}".format( ( se ) )+'.h5'  # Weight file
        weifile = 'smallcnn'+"{0:03d}".format( ( ep ) )+'.h5'   # Weight file
        #weifile = 'smallcnn'+"{0:03d}".format( ( 0 ) )+'.h5'   # Weight file
        #model.save_weights( os.path.join( "F:/work/dev/_PROGRES/python20190601/__log/", weifile ) )
        model.save_weights( os.path.join( ('H:/temp/'+LOGNumb+'/'), weifile ) )
        times   = time.time() - start # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='' )
        print( " [VGG-]:%6.2f[%%]   [LFW-]:%6.2f[%%]" %( 100.0*precT, 100.0*precV ), flush=True, end='\n' )
        
        #----------------------------------------------------------------------
        #
        #  Check processing time and clear session
        #
        #----------------------------------------------------------------------
        times   = time.time() - epset # Processing time
        with open( logTime, "a" ) as f_handle:
            f_handle.write( "%d\t%5d[min]\t%3.1f[h]\thuman:%10d\n" %(ep,(int)((times+0.5)/60.0),times/3600.0,human) ) 
        
        #----------------------------------------------------------------------
        # Jump here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #----------------------------------------------------------------------
        #label .END
        for se in range( 1 ):
            K.clear_session ()
            gc.collect      ()    
            
    #--------------------------------------------------------------------------
    #
    #  Clear session and terminate training
    #
    #--------------------------------------------------------------------------
    #del( trvgg_x )
    #del( trvgg_y )
    del( valfw_x )
    del( valfw_y )
    del( vavgg_x )
    del( vgg_bio )
    gc.collect  ()
    K.clear_session()
    
    #--------------------------------------------------------------------------
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #--------------------------------------------------------------------------

# End of files ----------------------------------------------------------------
