from sklearn.ensemble import RandomForestClassifier
import numpy as np

traindata=[]
testdata={}
class Domain:
    def __init__(self,_name,_label,_length,numm):
        self.name=_name
        self.label=_label
        self.length=_length
        self.numm=numm

    def returndata(self):
        return [self.length,self.numm]

    def returnlabel(self):
        if self.label=="dga":
            return 1
        else:
            return 0

def init_Data(filename):
    with open(filename) as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line=="":
                continue
            tokens=line.split(",")
            name=tokens[0]
            label=tokens[1]
            length=len(name)
            countnum=0
            for i in name:
                if i.isdigit():
                    numm+=1
            traindata.append(Domain(name,label,length,numm))

def init_test(filename):
    with open(filename) as m:
        for line in m:
            length=len(line)
            numm=0
            for j in line:
                if j.isdigit():
                   numm+=1
            testdata[line]=[length,numm]

def main():
    init_Data("train.txt")
    init_test("test.txt")
    trainmatrix=[]
    labellist=[]
    for item in traindata:
        trainmatrix.append(item.returndata())
        labellist.append(item.returnlabel())

    clf=RandomForestClassifier(random_state=0)
    clf.fit(trainmatrix,labellist)
    doc=open("result.txt","w")
    for item2 in testdata.items():
        temp=clf.predict([item2[1]])
        if temp==1:
            label='dga'
        else:
            label='notdga'
        print(item2[0][:-1],label,sep=',',file=doc)
    doc.close()

if __name__=='__main__':
    main()