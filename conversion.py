#Dependencies
import struct
import idx2numpy
import pickle
import numpy

#Defining how to write into a file using pickling
def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()

#Defining how to read a file using pickles
def read(filename):
        f = open(filename)
        data = pickle.load(f)
        f.close()
        return data

#Defining the conversion function
def conversion(document,new_file):

    #Determining the size and magic of the document for error checking
    with open(document, 'rb') as f:
        bytes = f.read(8)
        magic, size = struct.unpack(">II", bytes)

        print(magic)
        print(size)

    #Setting the output of this function as an array
    ndarr = idx2numpy.convert_from_file(document)
    print(ndarr)

    #
    f_read = open(document, 'rb')
    print(f_read)

    ndarr = idx2numpy.convert_from_file(f_read)
    print(ndarr)

    write(ndarr, new_file)
    read_data = read(new_file)

    print(read_data)

if __name__ == "__main__":
    conversion('/Users/suhahussain/Documents/MachineLearning/train-images-idx3-ubyte (1)','train-images-idx3.txt')
    conversion('/Users/suhahussain/Documents/MachineLearning/train-labels-idx1-ubyte','train-labels-idx1.txt')
    conversion('/Users/suhahussain/Documents/MachineLearning/t10k-images-idx3-ubyte','t10k-images-idx3.txt')
    conversion('/Users/suhahussain/Documents/MachineLearning/t10k-labels-idx1-ubyte','t10k-labels-idx1.txt')
