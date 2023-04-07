## 評価関数
def fp(B,C,D,E,G,H,I,J,L,M,N,O):
    FP0=B+C+D
    FP1=E+G+H
    FP2=I+J+L
    FP3=M+N+O
    return FP0+FP1+FP2+FP3

def fn(E,I,M,B,J,N,C,G,O,D,H,L):
    FN0=E+I+M
    FN1=B+J+N
    FN2=C+G+O
    FN3=D+H+L
    return FN0+FN1+FN2+FN3

def tn(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P):
    TN0=F+G+H+J+K+L+N+O+P
    TN1=A+C+D+I+K+L+M+O+P
    TN2=A+B+D+E+F+H+M+N+P
    TN3=A+B+C+E+F+G+I+J+K
    return TN0+TN1+TN2+TN3

def tp(A,F,K,P):
    return A+F+K+P

def accuracy(TP,data_size):
    return TP/data_size

def precision(TP,FP):
    if TP+FP == 0:
        return 0
    else:
        return TP/(TP+FP)

def recall(TP,FN):
    if TP+FN == 0:
        return 0
    else:
        return TP/(TP+FN)

def tpr(TP,FN):
    if TP+FN == 0:
        return 0
    else:
        return TP/(TP+FN)

def fpr(FP,TN):
    if FP+FP == 0:
        return 0
    else:
        return FP/(TN+FP)