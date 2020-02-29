###############################################################################
#
#   Image Recognition with CNN (using PyTorch libraries)
# 
#		Copyright	:: BLVRBIRD
#		Date        :: 2020.01.30
#		CPU         :: **
#		Language    :: Python
#		File Name   :: python_trainVGG_softmax0.py
#
###############################################################################
WORK_DIRECTORY  = 'F:/work/dev/_PROGRES/python20191110' # Work directory
LIST_VGG        = '/listTrainVggAll.txt'
FILE_SUM        = '/fileSumAll.txt'
CLASS_NUM       = 8547
RNDSeed         = 0                                     # Change here!!!!
PRENumb         = '0010'                                # Change here!!!!
LOGNumb         = '0011'                                # Change here!!!!

#------------------------------------------------------------------------------
#  General Library
#------------------------------------------------------------------------------
import os
os.chdir( WORK_DIRECTORY )                              # Change Work Directory
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                # See https://qiita.com/ballforest/items/3f21bcf34cba8f048f1e
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # See https://qiita.com/tanemaki/items/e3daa1947a34f63ab6e3
CUDA_LAUNCH_BLOCKING=1                                  # See https://qiita.com/takumiabe/items/fd6855737b08cd6c7612
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
import cloudpickle
import cv2 
import numpy                        as np
import pandas                       as pd
import gc
import io
import sys
from   PIL                          import Image
from   scipy                        import *
from   pylab                        import *
from   shutil                       import copyfile

#------------------------------------------------------------------------------
#  Tensorflow
#------------------------------------------------------------------------------
import tensorflow as tf
from   tensorflow.python.client     import device_lib
device_lib.list_local_devices()
config                              = tf.ConfigProto()
config.gpu_options.allow_growth     = True
session                             = tf. Session(config=config)

#------------------------------------------------------------------------------
#  PyTorch
#------------------------------------------------------------------------------
import torch
from   torch.autograd               import Variable                     # Auto Grad Calc
import torch.nn                     as nn                               # nn for Network
from   torch.nn.parameter           import Parameter                    # nn.parameter
import torch.optim                  as optim                            # Optimizer
import torch.nn.functional          as F                                # Network util
import torch.utils.data                                                 # Dataset read
import torchvision                                                      # Vision
from   torchvision                  import datasets, models, transforms # Dataset etc.
import torchvision.models           as models
import pretrainedmodels             # print( pretrainedmodels.model_names)
import torch.backends               as tb
tb.cudnn.enabled                    = False # https://blog.csdn.net/weixin_38673554/article/details/103022918
tb.cudnn.deterministic              = True  # https://qiita.com/chat-flip/items/c2e983b7f30ef10b91f6
gpu_id                              = 0     # http://kaga100man.com/2019/12/08/post-110/
if torch.cuda.is_available():
    device                          = torch.device( f'cuda:{gpu_id}' )
else:
    device                          = torch.device( 'cpu' )

#------------------------------------------------------------------------------
#
#  Global valiables, buffers, paths
#
#------------------------------------------------------------------------------
bach_size       =  256          # or 128
bacv_size       =  250          # or 100
dim_size        =  512          # 128
targ_size       =   96          #  96
targ_siz2       =   96          #  96
chan_size       =    3
#------------------------------------------------------------------------------
max_degree      =    8
max_resize      = 0.08
#------------------------------------------------------------------------------
Lrate           = 0.01000       # learning rate
Alpha           = 0.00020       # weight decay 2e-4
Scale           =  16.0         # Scale factor
nb_epoch        = 2000          # epoch number
nb_steps        =  200          # steps/epoch 100,200
nb_dots         = nb_steps//10  # nb_steps/10
Valid0          = 5000          # VGGtest test num
Valid1          = 1024          # Softmax test num
#------------------------------------------------------------------------------
pp              = [ 0,300,600,900,1200,1500,1800,2100,2400,2700,3000 ]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
#  https://qiita.com/elm200/items/46633430c456dd90f1e3
#
#------------------------------------------------------------------------------
def use_gpu( e ):
    if torch.cuda.is_available():
        return e.cuda()
    return ( e )

#------------------------------------------------------------------------------
#
#  See https://github.com/Shimpei-GANGAN/metric_learning_fastai/blob/master/main.py
#
#------------------------------------------------------------------------------
class L2ConstraintedNet( nn.Module ): 
    def __init__( self, org_model, alpha=Scale, num_classes=CLASS_NUM ):
        super( L2ConstraintedNet, self ).__init__() 
        self.Mnetv2 = org_model
        self.alpha1 = alpha
        self.drpout = nn.Dropout( p=0.2, inplace=False )
        self.conv2d = nn.Conv2d ( 1280, dim_size, kernel_size=(3,3), stride=(1,1), bias=False )
        self.dense1 = nn.Linear ( dim_size, num_classes, bias=True )
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        
    def fwd_train( self, x ):
        # This exists since TorchScript doesn't support inheritance, so the superclass method 
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass 
        x           = self.Mnetv2.features ( x )
        x           = self.drpout ( x )
        x           = self.conv2d ( x )                         # Convolution downsize to 1x1 [N,1280,1,1]
        x           = x.view( x.size(0), x.size(1) )            # x = x.mean( [2,3] ) also OK
        l2          = torch.norm( x, dim=1, keepdim=True )
        l2_         = 1.0/(l2+1e-10)
        x           = self.alpha1  * x * l2_                    # multiply alpha
        x           = self.dense1  ( x )                        # classification
        x           = F.log_softmax( x, dim=-1 )                # softmax
        return ( x )
    
    def fwd_valid( self, x ): 
        x           = self.Mnetv2.features ( x )
        x           = self.conv2d ( x )                         # Convolution downsize to 1x1 [N,1280,1,1]
        x           = x.view( x.size(0), x.size(1) )            # x = x.mean( [2,3] ) also OK
        l2          = torch.norm( x, dim=1, keepdim=True )
        l2_         = 1.0/(l2+1e-10)
        x           = x * l2_                                   # L2 Normalize
        return ( x )
    
    def forward( self, x ):
        return self.fwd_train( x )

#------------------------------------------------------------------------------
#
#  Set Random Data Sequence
#
#------------------------------------------------------------------------------
def set_random_data                 ( img                   ,   # not array
                                      flag                  ,   # 0:train, 1:valid
                                      loop                  ,
                                      t_v_x                 ,
                                      t_v_y                 ,
                                      a                     ):
    imgarry = np.array( img )
    t_v_x.append(imgarry)
    t_v_y.append(   a   )
    a       = a+1
    #if flag==1:
    return t_v_x, t_v_y, a

###############################################################################
###############################################################################
###############################################################################
###############################################################################
#------------------------------------------------------------------------------
#
#
#  M A I N  F U N C T I O N
#
#
#------------------------------------------------------------------------------
###############################################################################
###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
    #
    #  Create LOG files
    #
    #--------------------------------------------------------------------------
    printm ( )
    DIRPath = WORK_DIRECTORY
    LOGPath = WORK_DIRECTORY + '/__log/' + LOGNumb + '/'        # Set log path
    logLoss = 'logLoss'+"{0:03d}".format( 0 )+'.txt'            # Set log path
    logLoss = os.path.join( LOGPath, logLoss )
    logLoss = logLoss.replace( "\\", "/" )
    logAccv = 'logAccv'+"{0:03d}".format( 0 )+'.txt'            # Set log path
    logAccv = os.path.join( LOGPath, logAccv )
    logAccv = logAccv.replace( "\\", "/" )
    logAcct = 'logAcct'+"{0:03d}".format( 0 )+'.txt'            # Set log path
    logAcct = os.path.join( LOGPath, logAcct )
    logAcct = logAcct.replace( "\\", "/" )
    logAccx = 'logAccx'+"{0:03d}".format( 0 )+'.txt'            # Set log path
    logAccx = os.path.join( LOGPath, logAccx )
    logAccx = logAccx.replace( "\\", "/" )
    logTime = 'logTime'+"{0:03d}".format( 0 )+'.txt'            # Set log path
    logTime = os.path.join( LOGPath, logTime )
    logTime = logTime.replace( "\\", "/" )
    logStat = 'logStat'+"{0:03d}".format( 0 )+'.txt'            # Set log path
    logStat = os.path.join( LOGPath, logStat )
    logStat = logStat.replace( "\\", "/" )
    open( logLoss, "w", encoding="utf-8").close                 # Just create
    open( logAccv, "w", encoding="utf-8").close                 # Just create
    open( logAcct, "w", encoding="utf-8").close                 # Just create
    open( logAccx, "w", encoding="utf-8").close                 # Just create
    open( logTime, "w", encoding="utf-8").close                 # Just create
    open( logStat, "w", encoding="utf-8").close                 # Just create

    #--------------------------------------------------------------------------
    #
    #  Open whole VGG image files
    #
    #--------------------------------------------------------------------------
    print( "Open and read target files.................", flush=True, end='\n' )
    start   = time.time() # Processing time
    #--------------------------------------------------------------------------
    listlin = Valid0                                            # 5,000
    listlin_= 1.0/listlin                                       # 1/5000
    fLO     = open((WORK_DIRECTORY+LIST_VGG))                   # Change here!!!!
    fSM     = open((WORK_DIRECTORY+FILE_SUM))                   # Change here!!!!
    lineLO  = fLO.readlines()                                   # Read 1 line as string(include \n)
    lineSM  = fSM.readlines()                                   # Read 1 line as string(include \n)
    fLO.close ()
    fSM.close ()
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
    a       = 0
    for path in lineLO:
        a   = a + 1
    vggnum  = a                                                 # Total image #
    a       = 0
    for path in lineSM:
        a   = a + 1
    dirnum  = a
    if( dirnum!=CLASS_NUM ):                                    # Total class #
        print( " dirnum unmatched ERROR!!!\n" )
        sys.exit()
    #--------------------------------------------------------------------------
    
    np.random.seed( RNDSeed )                                   # random seed
        
    #--------------------------------------------------------------------------
    #
    #  Set VGG validation data
    #
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
        img_pil = Image.open  ( vgg_bio[Anc] ) # open image
        imgarry = np.array    ( img_pil      ) # img_to_array( img_pil      )
        vaanc.append( imgarry )
        img_pil = Image.open  ( vgg_bio[Pos] ) # open image
        imgarry = np.array    ( img_pil      ) # img_to_array( img_pil      )
        vapos.append( imgarry )
        while( 1 ): # Negative
            pr  = np.random.randint ( dirnum ) # Negative ID
            r1  = int(lineSM[pr])
            if( pr!=ar ):
                if( pr>0 ): r0 = int(lineSM[pr-1])
                else      : r0 = 0
                Neg     = np.random.randint( r0, r1 ) # File #
                break
        img_pil = Image.open  ( vgg_bio[Neg] ) # open image
        imgarry = np.array    ( img_pil      ) # img_to_array( img_pil      )
        vaneg.append( imgarry )  
    vavgg_x = [] # append img_to_array(img)
    for L in range( listlin ):
        vavgg_x.append( np.array( vaanc[L] ) ) # img_to_array( vaanc[L] ) )
    for L in range( listlin ):
        vavgg_x.append( np.array( vapos[L] ) ) # img_to_array( vapos[L] ) )
    for L in range( listlin ):
        vavgg_x.append( np.array( vaneg[L] ) ) # img_to_array( vaneg[L] ) )
    vavgg_x = np.array( vavgg_x[0:] )                           # Important!!
    vavgg_x = vavgg_x.astype('float32')                         # Cast to float32
    vavgg_x = torch.from_numpy(((vavgg_x-127.5)/127.5))         # Normalize[-1~1]
    vavgg_x = vavgg_x.transpose( 2, 3 )                         # num，ch，hei，wid
    vavgg_x = vavgg_x.transpose( 1, 2 )                         # num，ch，hei，wid
    del vaanc
    del vapos
    del vaneg
    gc.collect  ()
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #  Set LFW validation data
    #
    #--------------------------------------------------------------------------
    fVA     = open( (WORK_DIRECTORY+'/list4LFW_A.txt') )        # Annotation New LFW
    fVB     = open( (WORK_DIRECTORY+'/list4LFW_B.txt') )        # Annotation New LFW
    fVC     = open( (WORK_DIRECTORY+'/list4LFW_C.txt') )        # Annotation New LFW
    fVD     = open( (WORK_DIRECTORY+'/list4LFW_D.txt') )        # Annotation New LFW
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
           #img     = load_img   ( path )
            img     = Image.open ( path ) # open image
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    a = b = 0
    for path in lineB:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
           #img     = load_img   ( path )
            img     = Image.open ( path ) # open image
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    a = b = 0
    for path in lineC:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
           #img     = load_img   ( path )
            img     = Image.open ( path ) # open image
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    a = b = 0
    for path in lineD:
        if b>=pp[0] and b<pp[10]:  
            path    = path.rstrip('\r\n')
           #img     = load_img   ( path )
            img     = Image.open ( path ) # open image
            valfw_x, valfw_y, a = set_random_data( img, 1, 0, valfw_x, valfw_y, a )
        b = b+1
    lfwnum  = a
    lfwnum_ = 1.0/a
    valfw_x = np.array(  valfw_x[0:]  )                         # Important!!
    valfw_x = valfw_x.astype('float32')                         # Cast to float32
    valfw_x = torch.from_numpy(((valfw_x-127.5)/127.5))         # Normalize[-1~1]
    valfw_x = valfw_x.transpose( 2, 3 )                         # num，ch，hei，wid
    valfw_x = valfw_x.transpose( 1, 2 )                         # num，ch，hei，wid
    #--------------------------------------------------------------------------
    times = time.time() - start # Processing time
    print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )           # 283.11[sec]
    print( "Number of total training   images [%8d]" %vggnum, flush=True )  # 136038
    print( "Number of total validation images [%8d]" %lfwnum, flush=True )  # 300,600,3000
    
    #--------------------------------------------------------------------------
    #
    #  Set parameters
    #  See http://kaga100man.com/2019/01/05/post-84/
    #
    #--------------------------------------------------------------------------
    epoch   = nb_epoch                                          # epoch#
    steps   = nb_steps                                          # steps/epoch
    tryme   = steps//2                                          # tryme not used
    trmin   = 1                                                 # training min val(>=2) not used
    human   = 0                                                 # RESET
    precV   = 0.00                                              # RESET
    precT   = 0.00                                              # RESET
    precX   = 0.00                                              # RESET
    vggn_   = 1.0/vggnum                                        # 1/vggnum
    # Torch transformer
    transfm = transforms.Compose( [transforms.ToTensor(),
                                   transforms.Lambda( lambda x: (2.0*x-1.0) )] ) # Normalize[-1~1]
    x_train = torch.zeros( bach_size, chan_size, targ_size, targ_size ) # RESET
    #--------------------------------------------------------------------------
    print( "=================================================================", flush=True, end='\n' )
    epset = time.time() # Processing time
    
    for ep in range( 0, epoch ):

        #----------------------------------------------------------------------
        #
        #  Create CN network model
        #
        #----------------------------------------------------------------------
        print( "Creating CN network model..................", flush=True, end='' )
        start = time.time() # Processing time
        #----------------------------------------------------------------------
        backbone    = torchvision.models.mobilenet_v2   ( pretrained=False )
        model       = L2ConstraintedNet                 ( backbone )
        if( ep==0 ):
            # Save weights
            torch.save( model.state_dict(), os.path.join( 'H:/temp', 'smallcnnINI.pth' ) )
            # See https://qiita.com/derodero24/items/f4cc46f144f404054501
            with open( 'H:/temp/mobilenet2.pkl', 'wb' ) as f: cloudpickle.dump( model, f )
        #----------------------------------------------------------------------
        times = time.time() - start                             # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )

        #----------------------------------------------------------------------
        #
        #  Get Weight data from h5 file
        #
        #----------------------------------------------------------------------
        print( "Get Weight data from pth file..............", flush=True, end='' )
        start       = time.time()                               # Processing time
        #----------------------------------------------------------------------
        if( ep==0 ):
            weifile     = 'smallcnnINI.pth'                             # Log file
            WEIGHT_PATH = os.path.join( ('H:/temp/'        ), weifile ).replace( "\\", "/" )
        else       :
            weifile     = 'smallcnn'+"{0:03d}".format( (ep-1) )+'.pth'  # Log file
            WEIGHT_PATH = os.path.join( ('H:/temp/'+LOGNumb), weifile ).replace( "\\", "/" )
        #----------------------------------------------------------------------
        L_rate          = 0.00100
        '''
        if  ( ep<  20 ): L_rate=0.01000
        elif( ep<  50 ): L_rate=0.00500
        elif( ep< 100 ): L_rate=0.00100
        elif( ep< 200 ): L_rate=0.00050
        elif( ep< 400 ): L_rate=0.00010
        elif( ep< 600 ): L_rate=0.00005
        else           : L_rate=0.00001
        '''
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        #  Set training parameters
        #
        #----------------------------------------------------------------------
        batch           = bach_size                             # RESET 256
        batcv           = bacv_size                             # RESET 250
        TRY             = 0                                     # RESET
        lossmea         = 0.0                                   # RESET
        verb            = 0                                     # RESET
        #----------------------------------------------------------------------
        model.load_state_dict( torch.load( WEIGHT_PATH ) )      # load training results
        optimizer       = optim.SGD ( model.parameters(), lr=L_rate, weight_decay=Alpha )
        criterion       = nn.NLLLoss( reduction='sum' )         # nn.MSELoss() nn.NLLLoss()
        model           = model.to( device )                    # Model on GPU
        model.train()                                           # Training mode
        #----------------------------------------------------------------------
        times           = time.time() - start                   # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )
        #----------------------------------------------------------------------
        print( "[%6d] %10d: "  %( ep, human ), flush=True, end='\n' )
        print( "Now training ( batch_size:%4d )." %(batch), flush=True, end='' )
        start           = time.time()                           # Processing time
        
        #======================================================================
        #
        #  Training with VGG image files
        #
        #======================================================================
        for se in range( 1000 ):                                # while( 1 )
            randv           = [np.random.randint(0,dirnum) for ii in range(batch)]
            for ii in range( batch ):                           # batch: 256
                ar          = randv[ii]                         # Anchor/Positive ID
                r1          = int( lineSM[ar] )                 # Get sum val.
                if( ar>0 ): r0 = int( lineSM[ar-1] )            # File# range r0~r1
                else      : r0 = 0                              # File# range r0~r1
                Anc         = np.random.randint ( r0, r1 )      # File# random    
                img_pil     = Image.open  ( vgg_bio[Anc] )      # open images
                imgarry     = np.array    ( img_pil      )      # numpy float
                x_train[ii] = transfm     ( imgarry      )      # Torch type
            #------------------------------------------------------------------
            onehot      = torch.LongTensor(randv)               # Not one-hot
            optimizer.zero_grad()                               # Init grad
            x_train     = x_train.to( device )                  # use_gpu( x_train )
            onehot      = onehot.to ( device )                  # use_gpu( onehot  )
            output      = model.fwd_train( x_train )            # Forward
            loss        = criterion ( output, onehot )          # Calc Loss
            loss.backward ()                                    # Calc Grad
            optimizer.step()                                    # Update parameters
            lossmea    += loss.item ()                          # Add Loss
            #------------------------------------------------------------------
            human       = human + batch                         # Increment human#
            TRY         = TRY + 1                               # Increment TRY#
            if( ((se+1)%nb_dots)==0 ):                          # nb_dots:depends on steps
                print( ".", flush=True, end='' )
            if( TRY>=steps ):
                lossmea = lossmea / steps                       # loss mean
                times   = time.time() - start                   # Processing time
                print( "done --- %8.2f[sec]" %(times), flush=True, end='\n' )
                break
        del( output  )
        del( onehot  )
        del( randv   )
        #----------------------------------------------------------------------
        
        #======================================================================
        #
        #  Predict validation images and save logfiles
        #
        #======================================================================
        print( "PREDICT VALIDATION INAGES..................", flush=True, end='' )
        start           = time.time()                           # Processing time
        model.eval()                                            # Prediction mode
        max_0           = 300                                   # RESET max range
        max_1           = max_0 - 1                             # RESET
        max_2           = 250                                   # RESET max range for gnuplot
        Reso            = 200.0                                 # Resolution
        Reso_           = 0.005                                 # Resolution
        #----------------------------------------------------------------------
        histpos         = [0]*max_0
        histneg         = [0]*max_0
        sum_pos         = [0]*max_0
        sum_neg         = [0]*max_0
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        # VGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftmaxVGGSoftma
        #
        #----------------------------------------------------------------------
        x_valid         = torch.zeros( Valid1, chan_size, targ_size, targ_size ) # RESET
        randv           = [np.random.randint(0,dirnum) for ii in range(Valid1)]
        for ii in range( Valid1 ):                              # Valid1:1024
            ar          = randv[ii]                             # Anchor/Positive ID
            r1          = int( lineSM[ar] )                     # Check sum value
            if( ar>0 ): r0 = int( lineSM[ar-1] )                # File# range r0~r1
            else      : r0 = 0                                  # File# range r0~r1
            Anc         = np.random.randint ( r0, r1 )          # File#
            img_pil     = Image.open  ( vgg_bio[Anc] )          # Open image
            imgarry     = np.array    ( img_pil      )          # numpy float
            x_valid[ii] = transfm     ( imgarry      )          # Torch type
        onehot          = torch.LongTensor(randv)
        with torch.no_grad():                                   # Stop update
            jj          = 0
            kk          = 0
            for ii in range( Valid1//batch )  :                 # 1024/256=4
                vbatch  = x_valid[jj:jj+batch,:]
                obatch  = onehot [jj:jj+batch  ]
                jj      = jj+batch
                vbatch  = vbatch.to( device )
                obatch  = obatch.to( device )
                pred    = model.fwd_train( vbatch )             # Prediction
                for mm in range( batch ):                       # batch:256
                    if( torch.argmax( pred[mm] )==obatch[mm] ):
                        kk = kk + 1                             # INCRE
        precX           = kk/Valid1                             # Accuracy
        del( x_valid )
        del( randv   )
        del( onehot  )
        del( obatch  )
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        # LFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFWLFW
        #
        #----------------------------------------------------------------------
        for ii in range( max_0 ): histpos[ii]=histneg[ii]=sum_pos[ii]=sum_neg[ii]=0 # RESET
        #----------------------------------------------------------------------
        predlfw         = torch.zeros( 0, dim_size )            # RESET
        predlfw         = predlfw.to( device )
        with torch.no_grad():                                   # Stop update
            jj          = 0
            kk          = 0
            for ii in range( 4*lfwnum//batcv ):                 # 4*3000/250=48
                vbatch  = valfw_x[jj:jj+batcv,:]                # dtype=torch.float64)
                jj      = jj+batcv
                vbatch  = vbatch.to( device )
                pred    = model.fwd_valid( vbatch )             # Prediction
                predlfw = torch.cat( [predlfw,pred], dim=0 )
        #----------------------------------------------------------------------        
        posiA   = predlfw[0                   :lfwnum,                     :]
        posiB   = predlfw[lfwnum              :lfwnum+lfwnum,              :]
        negaC   = predlfw[lfwnum+lfwnum       :lfwnum+lfwnum+lfwnum,       :]
        negaD   = predlfw[lfwnum+lfwnum+lfwnum:lfwnum+lfwnum+lfwnum+lfwnum,:]
        #----------------------------------------------------------------------
        posidis = torch.bmm( posiA.unsqueeze(1), posiB.unsqueeze(2) ).squeeze(2)[:lfwnum]
        negadis = torch.bmm( negaC.unsqueeze(1), negaD.unsqueeze(2) ).squeeze(2)[:lfwnum]
        disposi = 'dis_posv'+"{0:03d}".format( ep )+'.txt'      # Log file Posi
        disnega = 'dis_negv'+"{0:03d}".format( ep )+'.txt'      # Log file Nega
        disposi = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disposi )
        disnega = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disnega )
        #----------------------------------------------------------------------
        pmean   = 0.0
        pvari   = 0.0
        nmean   = 0.0
        nvari   = 0.0
        for ii in  range( lfwnum ):    
            kk   = posidis[ii].item() * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histpos[jj]     = histpos[jj] + 1
            pmean = pmean + posidis[ii].item()
            pvari = pvari +(posidis[ii].item()*posidis[ii].item())
            kk   = negadis[ii].item() * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histneg[jj]     = histneg[jj] + 1
            nmean = nmean + negadis[ii].item()
            nvari = nvari +(negadis[ii].item()*negadis[ii].item())
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
        sum_pos[0]      = histpos[0]
        sum_neg[0]      = histneg[0]
        for ii in range( max_0 ):
            if( ii > 0 ):
                sum_pos[ii] = sum_pos[ii-1]+histpos[ii]
                sum_neg[ii] = sum_neg[ii-1]+histneg[ii]
        jj = kk = 0
        for ii in range( max_0 ):
            if( sum_pos[ii]>=(lfwnum-sum_neg[ii]) ):
                jj=ii
                if( sum_pos[ii]==(lfwnum-sum_neg[ii]) ):
                    kk      = jj
                break
        if( kk==0 ) : kk    = jj - 1
        else        : kk    = jj
        if( kk< 0 ) : kk    = 0
        pp = qq = 0
        for ii in range( jj+1 ): pp  = pp + histpos[ii]
        for ii in range( kk+1 ): qq  = qq + histpos[ii]
        if( qq> 0 ) : pp    = (pp+qq)/2 # check
        precV   = (lfwnum-pp) / lfwnum
        del( predlfw )
        del( posidis )
        del( negadis )
        del( posiA   )
        del( posiB   )
        del( negaC   )
        del( negaD   )
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        # VGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGGVGG
        #
        #----------------------------------------------------------------------
        for ii in range( max_0 ): histpos[ii]=histneg[ii]=sum_pos[ii]=sum_neg[ii]=0 # RESET
        #----------------------------------------------------------------------
        predvgg         = torch.zeros( 0, dim_size )            # RESET
        predvgg         = predvgg.to( device )
        with torch.no_grad():                                   # Stop update
            jj          = 0
            kk          = 0
            for ii in range( 3*listlin//batcv ):                # 3*5000/250=60
                vbatch  = vavgg_x[jj:jj+batcv,:]                # dtype=torch.float64)
                jj      = jj+batcv
                vbatch  = vbatch.to( device )
                pred    = model.fwd_valid( vbatch )             # Prediction
                predvgg = torch.cat( [predvgg,pred], dim=0 )
        #----------------------------------------------------------------------        
        posiA   = predvgg[0              :listlin,                :]
        posiB   = predvgg[listlin        :listlin+listlin,        :]
        negaC   = predvgg[listlin+listlin:listlin+listlin+listlin,:]
        #----------------------------------------------------------------------
        posidis = torch.bmm( posiA.unsqueeze(1), posiB.unsqueeze(2) ).squeeze(2)[:listlin]
        negadis = torch.bmm( posiA.unsqueeze(1), negaC.unsqueeze(2) ).squeeze(2)[:listlin]
        disposi = 'disTposv'+"{0:03d}".format( ep )+'.txt'      # Log file Posi
        disnega = 'disTnegv'+"{0:03d}".format( ep )+'.txt'      # Log file Nega
        disposi = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disposi )
        disnega = os.path.join( (WORK_DIRECTORY+"/__log/"+LOGNumb+'/'), disnega )
        #----------------------------------------------------------------------
        pmean   = 0.0
        pvari   = 0.0
        nmean   = 0.0
        nvari   = 0.0
        for ii in  range( listlin ):    
            kk   = posidis[ii].item() * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histpos[jj]     = histpos[jj] + 1
            pmean = pmean + posidis[ii].item()
            pvari = pvari +(posidis[ii].item()*posidis[ii].item())
            kk   = negadis[ii].item() * Reso
            if   kk<    0: jj = int(     0 )
            elif kk>max_1: jj = int( max_1 )
            else         : jj = int( kk + 0.5 )
            histneg[jj]     = histneg[jj] + 1
            nmean = nmean + negadis[ii].item()
            nvari = nvari +(negadis[ii].item()*negadis[ii].item())
        pmean   = pmean*listlin_
        pvari   = pvari*listlin_
        pvari   = pvari-pmean*pmean
        nmean   = nmean*listlin_
        nvari   = nvari*listlin_
        nvari   = nvari-nmean*nmean
        #----------------------------------------------------------------------
        with open( disposi, "w" ) as f_handle:
            for ii in range( max_2 ):  f_handle.write( "%lf\t%.6f\n" %( Reso_*ii, histpos[ii] ) )
        with open( disnega, "w" ) as f_handle:
            for ii in range( max_2 ):  f_handle.write( "%lf\t%.6f\n" %( Reso_*ii, histneg[ii] ) )
        with open( logStat, "a" ) as f_handle:
            f_handle.write( "( %.3f, %.3f ),( %.3f, %.3f )\n" %( pmean, pvari, nmean, nvari ) )
        #----------------------------------------------------------------------
        # Calculate accuraccy
        #----------------------------------------------------------------------
        sum_pos[0]      = histpos[0]
        sum_neg[0]      = histneg[0]
        for ii in range( max_0 ):
            if( ii > 0 ):
                sum_pos[ii] = sum_pos[ii-1]+histpos[ii]
                sum_neg[ii] = sum_neg[ii-1]+histneg[ii]
        jj = kk = 0
        for ii in range( max_0 ):
            if( sum_pos[ii]>=(listlin-sum_neg[ii]) ):
                jj=ii
                if( sum_pos[ii]==(listlin-sum_neg[ii]) ):
                    kk      = jj
                break
        if( kk==0 ) : kk    = jj - 1
        else        : kk    = jj
        if( kk< 0 ) : kk    = 0
        pp = qq = 0
        for ii in range( jj+1 ): pp  = pp + histpos[ii]
        for ii in range( kk+1 ): qq  = qq + histpos[ii]
        if( qq> 0 ) : pp    = (pp+qq)/2 # check
        precT   = (listlin-pp) / listlin
        del( predvgg )
        del( posidis )
        del( negadis )
        del( posiA   )
        del( posiB   )
        del( negaC   )
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        #
        #  Save training log information
        #
        #----------------------------------------------------------------------
        EPOCH = human * vggn_
        with open( logAccx, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, precX   ) )
        with open( logAccv, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, precV   ) )
        with open( logAcct, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, precT   ) )
        with open( logLoss, "a" ) as f_handle:
            f_handle.write( "%lf\t%.4f\n" %( EPOCH, lossmea ) )
        #----------------------------------------------------------------------
        times           = time.time() - start                   # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='' )
        print( " [LOSS]:%11.4f [ACCU]:%6.2f[%%]" %( lossmea, 100.0*precX ), flush=True, end='\n' )
        
        #----------------------------------------------------------------------
        #
        #  Save parameters to data file smallcnnXXX.h5
        #
        #----------------------------------------------------------------------
        print( "Save parameters to data file...............", flush=True, end='' )
        start           = time.time()                                   # Processing time
        weifile         = 'smallcnn'+"{0:03d}".format( ( ep ) )+'.pth'  # Save weights
        torch.save( model.state_dict(), os.path.join( ('H:/temp/'+LOGNumb+'/'), weifile ) )
        times           = time.time() - start                           # Processing time
        print( "done --- %8.2f[sec]" %(times), flush=True, end='' )
        print( " [VGG-]:%6.2f[%%]   [LFW-]:%6.2f[%%]" %( 100.0*precT, 100.0*precV ), flush=True, end='\n' )
        
        #----------------------------------------------------------------------
        #
        #  Check processing time and clear session
        #
        #----------------------------------------------------------------------
        times   = time.time() - epset                           # Processing time
        with open( logTime, "a" ) as f_handle:
            f_handle.write( "%d\t%5d[min]\t%3.1f[h]\thuman:%10d\n" %(ep,(int)((times+0.5)/60.0),times/3600.0,human) ) 

        #----------------------------------------------------------------------
        #
        #  gc.collect
        #
        #----------------------------------------------------------------------
        del( vbatch  )
        del( pred    )
        del( histpos )
        del( histneg )
        del( sum_pos )
        del( sum_neg )
        gc.collect  ()    
            
    #--------------------------------------------------------------------------
    #
    #  Clear session and terminate training
    #
    #--------------------------------------------------------------------------
    del( x_train )
    del( valfw_x )
    del( vavgg_x )
    del( vgg_bio )
    gc.collect  ()
    torch.cuda.empty_cache()
    
# End of files ----------------------------------------------------------------
