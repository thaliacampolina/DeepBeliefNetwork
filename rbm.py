import numpy

#Data is divided into lines(numcases), colums(numdims)
#Batches = number of examples
#Numhid = number of hidden labels
#Vishid = weight from visible to hidden layer

class RBM:
    def __init__(self, data, numhid): #new RBM => restart=1
        self.epoch = 1 #epoch counter
        self.numcases, self.numdims, self.numbatches = data.shape
        self.vishid    = 0.1*numpy.random.randn(self.numdims, numhid)
        self.hidbiases = numpy.zeros((1, numhid))
        self.visbiases = numpy.zeros((1, self.numdims))

        self.poshidprobs = numpy.zeros((self.numcases, numhid));
        self.neghidprobs = numpy.zeros((self.numcases, numhid));
        self.posprods    = numpy.zeros((self.numdims, numhid));
        self.negprods    = numpy.zeros((self.numdims, numhid));
        self.vishidinc   = numpy.zeros((self.numdims, numhid));
        self.hidbiasinc  = numpy.zeros((1, numhid));
        self.visbiasinc  = numpy.zeros((1, self.numdims));
        self.batchposhidprobs = numpy.zeros((self.numcases, numhid, self.numbatches));


    def __repr__(self):
        return  "Matrix: %s \n Data: %s %s %s"%(self.vishid, self.numcases, self.numdims, self.numbatches)



if __name__ == "__main__":
    data = numpy.ones((2,3,7));
    numhid = 3
    rbm = RBM(data,numhid)
    print rbm
