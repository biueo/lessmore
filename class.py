import  os,shutil,re

scene='office'

rootpath = './%s/' %scene
if not os.path.exists(rootpath):
    print(filename+'not found')
    os._exit(0)
if not os.path.exists('./%s/train/'%scene):
    os.makedirs('./%s/train/rgb'%scene)
    os.makedirs('./%s/train/depth'%scene)
    os.makedirs('./%s/train/poses'%scene)
if not os.path.exists('./%s/test/'%scene):
    os.makedirs('./%s/test/rgb'%scene)
    os.makedirs('./%s/test/depth'%scene)
    os.makedirs('./%s/test/poses'%scene)

trainf = rootpath+'train/'
testf = rootpath+'test/'

with open ('./%s/TrainSplit.txt' %scene,'r') as f:
    trainfolders=f.readlines()
with open ('./%s/TestSplit.txt' %scene,'r') as f:
    testfolders=f.readlines()
trainorders=[re.search('\d{1,2}',i).group() for i in trainfolders]
testorders=[re.search('\d{1,2}',i).group() for i in testfolders]
#train files extract
seq_length=0
for seq in trainorders:
    seqfolder = './%s/seq-%s/' %(scene,format(int(seq),'02'))
    for data in os.listdir(seqfolder):
        label = data.split('.')[1]
        srcdata = seqfolder + data
        try:
            datanum = int(re.search('\d+',data).group())
        except:
            continue
        finalname = re.sub('\d+',format(datanum+seq_length,'06'),data)
        if label == 'color':
            objdata = trainf+'rgb/'+finalname
        elif label == 'depth':
            objdata = trainf + 'depth/'+finalname
        else:
            objdata = trainf + 'poses/'+finalname
        shutil.copy(srcdata,objdata)
    seq_length = len(os.listdir(seqfolder))/3+seq_length

# extract test files
seq_length=0
for seq in testorders:
    seqfolder = './%s/seq-%s/' %(scene,format(int(seq),'02'))
    for data in os.listdir(seqfolder):
        label = data.split('.')[1]
        srcdata = seqfolder + data
        datanum = int(re.search('\d+',data).group())
        finalname = re.sub('\d+',format(datanum+seq_length,'06'),data)
        if label == 'color':
            objdata = testf+'rgb/'+finalname
        elif label == 'depth':
            objdata = testf + 'depth/'+finalname
        else:
            objdata = testf + 'poses/'+finalname
        shutil.copy(srcdata,objdata)
    seq_length = len(os.listdir(seqfolder))/3+seq_length





