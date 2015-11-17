import os
import urllib
import utils
import gzip

def get_filenames():
    return [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
    ]

def ensure_datadir_exists():
    datadir = utils.datafile('')
    if os.path.exists(datadir):
        print 'Directory {0} already exists; skipping'.format(datadir)
    else:
        print 'Creating directory {0}'.format(datadir)
        os.makedirs(datadir)

def download_files():
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    for fn in get_filenames():
        file_url = base_url+fn+'.gz'
        local_path = utils.datafile(fn+'.gz')
        print 'Saving {0} to {1}'.format(file_url, local_path)
        urllib.urlretrieve(file_url,local_path)

def extract_files():
    for fn in map(utils.datafile, get_filenames()):
        with gzip.open(fn+'.gz', 'rb') as gzf:
            with open(utils.datafile(fn), 'wb') as f:
                print 'Extracting {0}'.format(gzf.name)
                f.write(gzf.read())
        
        print 'Deleting {0}.gz'.format(fn)
        os.remove(fn+'.gz')

# http://pjreddie.com/projects/mnist-in-csv/
def convert(imgf, labelf, outf, n):
    f = open(imgf, 'rb')
    o = open(outf, 'w')
    l = open(labelf, 'rb')

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(','.join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def convert_files():
    print 'Converting files to CSV...'
    filenames = map(utils.datafile, get_filenames())
    
    convert(filenames[0], filenames[1], utils.datafile('mnist-train.csv'), 60000)
    convert(filenames[2], filenames[3], utils.datafile('mnist-test.csv'), 10000)
    
    for fn in filenames:
        os.remove(fn)

def main():
    ensure_datadir_exists()
    download_files()
    extract_files()
    convert_files()
    

if __name__ == '__main__':
    main()
