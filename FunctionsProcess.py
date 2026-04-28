import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.metrics import auc
import pandas as pd
import numpy as np
import os
from ipywidgets import interact, widgets, fixed, interactive, HBox, Layout
import matplotlib as mpl


def getMetaDat(foldPath, foldNum):
    rot = []
    with open(foldPath+'/'+foldNum+'/pdata/1/reco', 'r+') as f:
        for line in f:
            if 'RECO_rotate' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                rot = float(next(f,''))
                break

    RECO_ft_mode = []
    with open(foldPath+'/'+foldNum+'/pdata/1/reco', 'r+') as f:
        for line in f:
            if 'RECO_ft_mode' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                RECO_ft_mode = next(f,'')
                break     

    bw = []
    with open(foldPath+'/'+foldNum+'/method', 'r+') as f:
        for line in f:
            if 'PVM_SpecSW=' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                bw = float(next(f,''))
                break


    bwHz = []
    with open(foldPath+'/'+foldNum+'/method', 'r+') as f:
        for line in f:
            if 'PVM_SpecSWH=' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                bwHz = float(next(f,''))
                break


    bwc = []
    with open(foldPath+'/'+foldNum+'/method', 'r+') as f:
        for line in f:
            if 'PVM_FrqWorkPpm' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(np.array(next(f,'')).astype(np.float))
                bwc = next(f,'')
                break
    bwc = float(bwc[0:bwc.index(' ')])

    ACQ_repetition_time = []
    with open(foldPath+'/'+foldNum+'/acqp', 'r+') as f:
        for line in f:
            if '##$ACQ_repetition_time=' in line.strip():
                ACQ_repetition_time = float(next(f,''))/1000
                break

    ACQ_Size = []
    with open(foldPath+'/'+foldNum+'/method', 'r+') as f:
        for line in f:
            if '##$PVM_SpecMatrix=' in line.strip():
                ACQ_Size = int(next(f,''))
                break

    NR = []
    with open(foldPath+'/'+foldNum+'/acqp', 'r+') as f:
        for line in f:
            if '##$NR=' in line.strip():
                # NR = float(next(f,''))
                NR = int(line.strip()[6:])
                break

    return(rot, RECO_ft_mode, bw, bwc, ACQ_repetition_time, ACQ_Size, NR, bwHz)




def ProcDat(foldPath, foldNum, NR, ACQ_Size, rot, bwc, bw, lbf):

    file_content = np.fromfile(foldPath+'/'+foldNum+'/pdata/1/fid_proc.64', np.float64)

    re = file_content[0::2]
    im = file_content[1::2]
    co = np.ndarray(len(re), dtype=np.complex128)


    for i in range(0,len(re)):
        co[i] = complex(re[i], im[i])
    
    co2 = np.reshape(co, (NR,ACQ_Size))
    co3 = np.reshape(co, (NR,ACQ_Size))

    lb = np.exp(-np.linspace(start=0, stop=lbf, num=len(co2[0])))
    for l in range(0,NR):
        co2[l] = co2[l]*lb



    RECO_rotate = [rot]
    dims = np.shape(co2)[1]
    phase_matrix = np.ones(dims)

    # for index in range(0,dims):
    f=np.array(range(dims))
    phase_vector=np.exp(complex(0,1)*2*np.pi*RECO_rotate[0]*f)

    ppms = np.linspace(bwc-bw/2, bwc+bw/2, dims)

    # fut = [fft(co2[i,:]*phase_vector) for i in range(np.shape(co2)[0])]
    fut = [fft(co2[i,:]*phase_vector) for i in range(np.shape(co2)[0])]
    futmag = [np.sqrt(np.real(fut[i])**2 + np.imag(fut[i])**2) for i in range(np.shape(co2)[0])]

    return(ppms, futmag, fut, co3)



def plotSingle(i, ppms, futmag, foldPath, foldNum):
    plt.figure(dpi=170)

    # plt.plot([184, 184],[min(futmag[i]), max(futmag[i])])
    plt.plot(ppms, futmag[i], linewidth = 0.5)
    plt.gca().invert_xaxis()
    plt.ylim(min(futmag[i])-(max(futmag[i])*0.1), max(futmag[i])+(max(futmag[i])*0.1))
    plt.savefig(foldPath+'/'+foldNum+'/SingleSpectra_NTP'+str(i)+'.png')
    plt.savefig(foldPath+'/'+foldNum+'/SingleSpectra_NTP'+str(i)+'.svg')
    
    plt.show()



def plot3D(tstep, NR, ppms, futmag, foldPath, foldNum):
    tim = list(range(0,NR*tstep, tstep))

    fig = plt.figure(dpi=200)
    ax = plt.axes(projection='3d')
    ax = plt.gca()

    for i in range(NR):
        ax.plot(ppms, futmag[i], np.zeros(np.size(ppms))+tim[i])

    ax.view_init( vertical_axis='y')

    ax.set_xlim(ax.get_xlim()[::-1])

    plt.savefig(foldPath+'/'+foldNum+'/AllSpectra3D.png')
    plt.savefig(foldPath+'/'+foldNum+'/AllSpectra3D.svg')
    
    plt.show()    


def integRegs(max_Lactate, min_Lactate, max_Alanine, min_Alanine, max_Piruvate, min_Piruvate, ppms, futmag, foldPath, foldNum, NR, zoomlow=[]):
        
    fig, ax = plt.subplots(figsize=(10, 7))
    if bool(zoomlow)==0:
        for i in range(NR):
            ax.plot(ppms, futmag[i])
    else:
        for i in range(NR):
            ax.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], futmag[i][abs(zoomlow[1]):abs(zoomlow[0])])


    ax.plot([min_Piruvate, min_Piruvate],[np.min(futmag), np.max(futmag)], color = 'black')
    ax.plot([max_Piruvate, max_Piruvate],[np.min(futmag), np.max(futmag)], color = 'black')

    ax.plot([min_Alanine, min_Alanine],[np.min(futmag), np.max(futmag)], color = 'black')
    ax.plot([max_Alanine, max_Alanine],[np.min(futmag), np.max(futmag)], color = 'black')

    ax.plot([min_Lactate, min_Lactate],[np.min(futmag), np.max(futmag)], color = 'black')
    ax.plot([max_Lactate, max_Lactate],[np.min(futmag), np.max(futmag)], color = 'black')


    ax.set_xlim(ax.get_xlim()[::-1])

    ax.fill_between(ppms, np.min(futmag), np.max(futmag), where= (ppms > min_Piruvate) & (ppms < max_Piruvate),
                    facecolor='red', alpha=0.2)

    ax.fill_between(ppms, np.min(futmag), np.max(futmag), where= (ppms > min_Alanine) & (ppms < max_Alanine),
                    facecolor='green', alpha=0.2)

    ax.fill_between(ppms, np.min(futmag), np.max(futmag), where= (ppms > min_Lactate) & (ppms < max_Lactate),
                    facecolor='yellow', alpha=0.2)
    ax.set_xlabel("ppm")
    plt.show()

    np.save(foldPath+'/'+foldNum+'/integRegTmp.npy', [min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate])


def plotInters(ppms,NR,futmag,foldPath,foldNum,plts=123):

    min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate = np.load(foldPath+'/'+foldNum+'/integRegTmp.npy')

    ppmSe = ppms[(ppms > min_Piruvate) & (ppms < max_Piruvate)]

    i=0
    inters = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Piruvate) & (ppms < max_Piruvate)]
        inters[i] = auc(ppmSe, intSe)


    ppmSe = ppms[(ppms > min_Alanine) & (ppms < max_Alanine)]

    i=0
    inters2 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Alanine) & (ppms < max_Alanine)]
        inters2[i] = auc(ppmSe, intSe)



    ppmSe = ppms[(ppms > min_Lactate) & (ppms < max_Lactate)]

    i=0
    inters3 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Lactate) & (ppms < max_Lactate)]
        inters3[i] = auc(ppmSe, intSe)

    plt.figure(dpi=150)

    valp = [int(d) for d in str(plts)]
    
    
    # Save data as CSV in main directory of experiment
    datT = np.transpose(pd.DataFrame([list(range(0, len(inters)*5, 5)), inters, inters2,inters3]))
    datT2 = datT.rename(columns={0: "Time", 1: "Pyruvate", 2: "Alanine", 3: "Lactate"})

    datT2.to_csv(foldPath+"/TemporalDataAll.csv", index=False, header=True)
    

    if plts==123:
        
        plt.scatter(list(range(0,NR*5, 5)), inters2, label="Alanine")
        plt.scatter(list(range(0,NR*5, 5)), inters3, label="Lactate")
        plt.scatter(list(range(0,NR*5, 5)), inters, label="Pyruvate")

    if len(valp)==2:
        if plts == 12:
            
            plt.scatter(list(range(0,NR*5, 5)), inters2, label="Alanine")
            plt.scatter(list(range(0,NR*5, 5)), inters, label="Pyruvate")
        elif plts == 13:
            
            plt.scatter(list(range(0,NR*5, 5)), inters3, label="Lactate")
            plt.scatter(list(range(0,NR*5, 5)), inters, label="Pyruvate")
        elif plts == 23:
            plt.scatter(list(range(0,NR*5, 5)), inters2, label="Alanine")
            plt.scatter(list(range(0,NR*5, 5)), inters3, label="Lactate")

    elif len(valp)==1:
        if plts == 1:
            plt.scatter(list(range(0,NR*5, 5)), inters, label="Pyruvate")
        elif plts == 2:
            plt.scatter(list(range(0,NR*5, 5)), inters2, label="Alanine")
        elif plts == 3:
            plt.scatter(list(range(0,NR*5, 5)), inters3, label="Lactate")

    # plt.yscale('log')
    plt.xlabel("Time (s)")
    leg = plt.legend(loc='upper right')
    plt.savefig(foldPath+'/'+foldNum+'/PlotInters.png')
    plt.savefig(foldPath+'/'+foldNum+'/PlotInters.svg')

    plt.show()

    return(inters,inters2,inters3)



def LacOverPyr(ppms, futmag, NR, foldPath, foldNum):
    min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate = np.load(foldPath+'/'+foldNum+'/integRegTmp.npy')

    ppmSe = ppms[(ppms > min_Piruvate) & (ppms < max_Piruvate)]

    i=0
    inters = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Piruvate) & (ppms < max_Piruvate)]
        inters[i] = auc(ppmSe, intSe)


    ppmSe = ppms[(ppms > min_Alanine) & (ppms < max_Alanine)]

    i=0
    inters2 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Alanine) & (ppms < max_Alanine)]
        inters2[i] = auc(ppmSe, intSe)



    ppmSe = ppms[(ppms > min_Lactate) & (ppms < max_Lactate)]

    i=0
    inters3 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Lactate) & (ppms < max_Lactate)]
        inters3[i] = auc(ppmSe, intSe)

    plt.figure(dpi=150)
    plt.scatter(list(range(0,NR*5, 5)), inters3/inters)

    plt.ylabel("Lactate/Pyruvate")

    plt.xlabel("Time (s)")
    plt.show()


def AlaOverPyr(ppms, futmag, NR, foldPath, foldNum):
    min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate = np.load(foldPath+'/'+foldNum+'/integRegTmp.npy')

    ppmSe = ppms[(ppms > min_Piruvate) & (ppms < max_Piruvate)]

    i=0
    inters = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Piruvate) & (ppms < max_Piruvate)]
        inters[i] = auc(ppmSe, intSe)


    ppmSe = ppms[(ppms > min_Alanine) & (ppms < max_Alanine)]

    i=0
    inters2 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Alanine) & (ppms < max_Alanine)]
        inters2[i] = auc(ppmSe, intSe)



    ppmSe = ppms[(ppms > min_Lactate) & (ppms < max_Lactate)]

    i=0
    inters3 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Lactate) & (ppms < max_Lactate)]
        inters3[i] = auc(ppmSe, intSe)

    plt.figure(dpi=150)
    plt.scatter(list(range(0,NR*5, 5)), inters2/inters)

    plt.ylabel("Alanine/Pyruvate")

    plt.xlabel("Time (s)")
    plt.show()


def LacOverAla(ppms, futmag, NR, foldPath, foldNum):
    min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate = np.load(foldPath+'/'+foldNum+'/integRegTmp.npy')

    ppmSe = ppms[(ppms > min_Piruvate) & (ppms < max_Piruvate)]

    i=0
    inters = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Piruvate) & (ppms < max_Piruvate)]
        inters[i] = auc(ppmSe, intSe)


    ppmSe = ppms[(ppms > min_Alanine) & (ppms < max_Alanine)]

    i=0
    inters2 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Alanine) & (ppms < max_Alanine)]
        inters2[i] = auc(ppmSe, intSe)



    ppmSe = ppms[(ppms > min_Lactate) & (ppms < max_Lactate)]

    i=0
    inters3 = np.zeros(NR)
    for i in range(NR):
        intSe = futmag[i][(ppms > min_Lactate) & (ppms < max_Lactate)]
        inters3[i] = auc(ppmSe, intSe)

    plt.figure(dpi=150)
    plt.scatter(list(range(0,NR*5, 5)), inters3/inters2)

    plt.ylabel("Lactate/Alanine")

    plt.xlabel("Time (s)")
    plt.show()


def plotStack(ts, ppms, futmag, foldPath, foldNum, zoomlow=[], multip = 1, ntp=[]):

    alldat = np.reshape(futmag, np.prod(np.shape(futmag)))
    futmagnorm = [(futmag[i]-min(alldat))/(max(alldat)-min(alldat)) for i in range(len(futmag))]

    plt.figure(dpi=174)
    if bool(ntp)==0 or ntp >= len(futmag):
        for i in range(len(futmag)):
            if bool(zoomlow)==0:
                plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
            else:
                plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmagnorm[i][abs(zoomlow[1]):abs(zoomlow[0])])*multip+i*ts, linewidth=0.5)
    else:
        for i in range(ntp):
            if bool(zoomlow)==0:
                plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
            else:
                plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmagnorm[i][abs(zoomlow[1]):abs(zoomlow[0])])*multip+i*ts, linewidth=0.5)
        #     plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
   
    
    
    plt.gca().invert_xaxis()
    plt.xlabel("ppm")
    plt.savefig(foldPath+'/'+foldNum+'/MNLikeStack.png')
    plt.savefig(foldPath+'/'+foldNum+'/MNLikeStack.svg')
    plt.show()


def integRegs2(Lactate, Alanine, Piruvate, ppms, futmag, foldPath, foldNum, NR, ts, multip,ntp, zoomlow=[]):
        

    alldat = np.reshape(futmag, np.prod(np.shape(futmag)))
    futmagnorm = [(futmag[i]-min(alldat))/(max(alldat)-min(alldat)) for i in range(len(futmag))]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    if bool(ntp)==0 or ntp >= len(futmag):
        for i in range(len(futmag)):
            if bool(zoomlow)==0:
                plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
            else:
                plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmagnorm[i][abs(zoomlow[1]):abs(zoomlow[0])])*multip+i*ts, linewidth=0.5)
    else:
        for i in range(ntp):
            if bool(zoomlow)==0:
                plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
            else:
                plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmagnorm[i][abs(zoomlow[1]):abs(zoomlow[0])])*multip+i*ts, linewidth=0.5)


    if bool(ntp)==0:
        mv=len(futmag)
    else: 
        mv=ntp

    ax.plot([abs(Piruvate[1]), abs(Piruvate[1])],[0, mv*ts], color = 'black')
    ax.plot([abs(Piruvate[0]), abs(Piruvate[0])],[0, mv*ts], color = 'black')

    ax.plot([abs(Alanine[1]), abs(Alanine[1])],[0, mv*ts], color = 'black')
    ax.plot([abs(Alanine[0]), abs(Alanine[0])],[0, mv*ts], color = 'black')

    ax.plot([abs(Lactate[1]), abs(Lactate[1])],[0, mv*ts], color = 'black')
    ax.plot([abs(Lactate[0]), abs(Lactate[0])],[0, mv*ts], color = 'black')


    ax.set_xlim(ax.get_xlim()[::-1])

    ax.fill_between(ppms, 0, mv*ts, where= (ppms > abs(Piruvate[1])) & (ppms < abs(Piruvate[0])),
                    facecolor='red', alpha=0.2)

    ax.fill_between(ppms, 0, mv*ts, where= (ppms > abs(Alanine[1])) & (ppms < abs(Alanine[0])),
                    facecolor='green', alpha=0.2)

    ax.fill_between(ppms, 0, mv*ts, where= (ppms > abs(Lactate[1])) & (ppms < abs(Lactate[0])),
                    facecolor='yellow', alpha=0.2)
        
    ax.set_xlabel("ppm")
    plt.show()

    np.save(foldPath+'/'+foldNum+'/integRegTmp.npy', [abs(Piruvate[1]),abs(Piruvate[0]),abs(Alanine[1]),abs(Alanine[0]),abs(Lactate[1]),abs(Lactate[0])])



def plotFID(i, foldPath, foldNum, co3):
    ACQ_Time = []
    with open(foldPath+'/'+foldNum+'/method', 'r+') as f:
        for line in f:
            if '##$PVM_SpecAcquisitionTime=' in line.strip():
                ACQ_Time = float(line.strip()[27:])
                break


    fig, (ax1, ax2) = plt.subplots(2,1, dpi=174)

    # plt.figure(dpi=174)
    ax1.plot(np.linspace(0,ACQ_Time,len(np.real(co3[i]))), np.real(co3[i]), linewidth=0.5, label='real', color='purple')
    # ax1.set_label('real')

    ax2.plot(np.linspace(0,ACQ_Time,len(np.imag(co3[i]))), np.real(co3[i]), linewidth=0.5, label='imag', color='red')
    ax2.set_xlabel('time (ms)')
    ax1.legend()
    ax2.legend()
    # plt.xlabel('time (ms)')
    plt.savefig(foldPath+'/'+foldNum+'/FID_NTP'+str(i)+'.png')
    plt.savefig(foldPath+'/'+foldNum+'/FID_NTP'+str(i)+'.svg')
    plt.show()



def integRegs3(BW, ppms, bwHz, futmag, ntp, zoomlow=[]):
            
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmag[ntp-1][abs(zoomlow[1]):abs(zoomlow[0])]), linewidth=0.5)


    ax.plot([abs(BW[1]), abs(BW[1])],[0, np.max(futmag[ntp-1])], color = 'black')
    ax.plot([abs(BW[0]), abs(BW[0])],[0, np.max(futmag[ntp-1])], color = 'black')

    ax.set_xlim(ax.get_xlim()[::-1])

    ax.fill_between(ppms, 0, np.max(futmag[ntp-1]), where= (ppms > abs(BW[1])) & (ppms < abs(BW[0])),
                    facecolor='red', alpha=0.2)
    
    ax.set_xlabel("ppm")
    plt.show()

    print('PPMs Width = '+str(abs(BW[0]) - abs(BW[1])))
    print('Hz Width = '+ str((bwHz/(max(ppms)-min(ppms)))*(abs(BW[0]) - abs(BW[1]))) + ' Hz')


def integRegs4(Noise, Peak, ppms, bwHz, futmag, ntp, zoomlow=[]):
            
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmag[ntp-1][abs(zoomlow[1]):abs(zoomlow[0])]), linewidth=0.5)


    ax.plot([abs(Noise[1]), abs(Noise[1])],[0, np.max(futmag[ntp-1])], color = 'black')
    ax.plot([abs(Noise[0]), abs(Noise[0])],[0, np.max(futmag[ntp-1])], color = 'black')

    ax.plot([abs(Peak[1]), abs(Peak[1])],[0, np.max(futmag[ntp-1])], color = 'black')
    ax.plot([abs(Peak[0]), abs(Peak[0])],[0, np.max(futmag[ntp-1])], color = 'black')



    ax.set_xlim(ax.get_xlim()[::-1])

    ax.fill_between(ppms, 0, np.max(futmag[ntp-1]), where= (ppms > abs(Noise[1])) & (ppms < abs(Noise[0])),
                    facecolor='red', alpha=0.2)
    ax.fill_between(ppms, 0, np.max(futmag[ntp-1]), where= (ppms > abs(Peak[1])) & (ppms < abs(Peak[0])),
                    facecolor='green', alpha=0.2)
    
    
    ax.set_xlabel("ppm")
    plt.show()


    pH = np.where(ppms == ppms[ppms > abs(Peak[0])][0])[0][0]
    pL = np.where(ppms == ppms[ppms > abs(Peak[1])][0])[0][0]

    nH = np.where(ppms == ppms[ppms > abs(Noise[0])][0])[0][0]
    nL = np.where(ppms == ppms[ppms > abs(Noise[1])][0])[0][0]


    SNR = np.max(futmag[ntp-1][pL:pH])/np.std(futmag[ntp-1][nL:nH])

    print('SNR = '+str(SNR))
    
    
def integralSumUp(ini, ene, ppms, futmag, foldPath, foldNum): # Clever way is tu sum the points you want in the curve...
    # i=5# Number of time points to consider for the sum

    Piruvate = np.zeros(2)
    Alanine = np.zeros(2)
    Lactate = np.zeros(2)

    min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate = np.load(foldPath+'/'+foldNum+'/integRegTmp.npy')
    Piruvate[1] = min_Piruvate
    Piruvate[0] = max_Piruvate
    Alanine[1] = min_Alanine
    Alanine[0] = max_Alanine
    Lactate[1] = min_Lactate
    Lactate[0] = max_Lactate

    plt.figure(dpi=170)
        # plt.plot([184, 184],[min(futmag[i]), max(futmag[i])])
    plt.plot(ppms, np.sum(futmag[ini-1:ene-1],0), linewidth = 0.5)
    plt.gca().invert_xaxis()
    plt.ylim(min(np.sum(futmag[ini-1:ene-1],0))-(max(np.sum(futmag[ini-1:ene-1],0))*0.1), max(np.sum(futmag[ini-1:ene-1],0))+(max(np.sum(futmag[ini-1:ene-1],0))*0.1))

    mv=100

    plt.plot([abs(Piruvate[1]), abs(Piruvate[1])],[0, max(np.sum(futmag[ini-1:ene-1],0))], color = 'black')
    plt.plot([abs(Piruvate[0]), abs(Piruvate[0])],[0, max(np.sum(futmag[ini-1:ene-1],0))], color = 'black')

    plt.plot([abs(Alanine[1]), abs(Alanine[1])],[0, max(np.sum(futmag[ini-1:ene-1],0))], color = 'black')
    plt.plot([abs(Alanine[0]), abs(Alanine[0])],[0, max(np.sum(futmag[ini-1:ene-1],0))], color = 'black')

    plt.plot([abs(Lactate[1]), abs(Lactate[1])],[0, max(np.sum(futmag[ini-1:ene-1],0))], color = 'black')
    plt.plot([abs(Lactate[0]), abs(Lactate[0])],[0, max(np.sum(futmag[ini-1:ene-1],0))], color = 'black')



    plt.fill_between(ppms, 0, max(np.sum(futmag[ini-1:ene-1],0)), where= (ppms > abs(Piruvate[1])) & (ppms < abs(Piruvate[0])),
                    facecolor='red', alpha=0.2)

    plt.fill_between(ppms, 0, max(np.sum(futmag[ini-1:ene-1],0)), where= (ppms > abs(Alanine[1])) & (ppms < abs(Alanine[0])),
                    facecolor='green', alpha=0.2)

    plt.fill_between(ppms, 0, max(np.sum(futmag[ini-1:ene-1],0)), where= (ppms > abs(Lactate[1])) & (ppms < abs(Lactate[0])),
                    facecolor='yellow', alpha=0.2)


    # plt.savefig(foldPath+'/'+foldNum+'/SumSpectra_NTP'+str(i)+'.png')
    # plt.savefig(foldPath+'/'+foldNum+'/SumSpectra_NTP'+str(i)+'.svg')

    plt.show()


    ppmSe = ppms[(ppms > min_Piruvate) & (ppms < max_Piruvate)]


    sumdat = np.sum(futmag[ini-1:ene-1],0)

    i=0
    inters = np.zeros(1)
    intSe = sumdat[(ppms > min_Piruvate) & (ppms < max_Piruvate)]
    inters = auc(ppmSe, intSe)

    ppmSe = ppms[(ppms > min_Alanine) & (ppms < max_Alanine)]

    i=0
    inters2 = np.zeros(1)
    intSe = sumdat[(ppms > min_Alanine) & (ppms < max_Alanine)]
    inters2 = auc(ppmSe, intSe)



    ppmSe = ppms[(ppms > min_Lactate) & (ppms < max_Lactate)]

    i=0
    inters3 = np.zeros(1)
    intSe = sumdat[(ppms > min_Lactate) & (ppms < max_Lactate)]
    inters3 = auc(ppmSe, intSe)

    nms = ["Pyr", "Ala", "Lac"]
    dts = [inters, inters2, inters3]
    df = pd.DataFrame([dts], columns = nms)

    df.to_csv(foldPath+"/SumUpIntegrals.csv", sep='\t')

    print('Pyruvate Integral: '+str(inters))
    print('Alanine  Integral: '+str(inters2))
    print('Lactate  Integral: '+str(inters3))
    
    return(inters, inters2, inters3)
    
def extrMetDat(foldPath):
    rot = []
    with open(foldPath+'/subject', 'r+') as f:
        for line in f:
            if 'SUBJECT_study_name' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                rot = str(next(f,''))
                break

    print('STUDY NAME: '+rot[1:-2])
    print('\n')
    
    rot = []
    with open(foldPath+'/subject', 'r+') as f:
        for line in f:
            if 'SUBJECT_dbirth' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                rot = str(next(f,''))
                break
            
    print('DATE BIRTH: '+rot[1:-2])
    print('\n')
    
    rot1 = []
    rot2 = []
    with open(foldPath+'/subject', 'r+') as f:
        for line in f:
            if 'SUBJECT_remarks' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                rot1 = str(next(f,''))
                rot2 = str(next(f,''))
                break

    print('REMARKS')
    print(rot1)
    print(rot2)
    print('\n')
    
    rot = []
    with open(foldPath+'/subject', 'r+') as f:
        for line in f:
            if 'SUBJECT_study_weight' in line.strip():
                # print(line.strip())
                # print(f.next())
                # print(next(f,''))
                rot = str(line.strip()[24:])
                break
            
    print('WEIGHT: '+rot)
    print('\n')
    
    
def integSelectScan(ppms, futmag, ntp):
            
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.plot(ppms, (futmag[ntp-1]), linewidth=0.5)

    ax.set_xlim(ax.get_xlim()[::-1])

    ax.set_xlabel("ppm")
    plt.show()

    
    
    
####################################################################################################################
# Function to extract automatically the experiment folder for 13C spectroscopy scan given a list of experiment directory paths
    
def getExpDirs(drsall):
    drsexp = []

    for i in range(len(drsall)):
        lsdrs = [os.listdir(drsall[i])[j] for j in range(len(os.listdir(drsall[i]))) if os.path.isdir(drsall[i]+'/'+os.listdir(drsall[i])[j])]
        
        lsdrtr = []

        for k in range(len(lsdrs)):
            try:
                if int(lsdrs[k]) < 150:
                    lsdrtr.append(lsdrs[k])
            except:
                pass
        
        for j in range(len(lsdrtr)):
            scn = []
            with open(drsall[i]+r'/'+lsdrtr[j]+r'\acqp', 'r+') as f:
                for line in f:
                    if 'ACQ_scan_name' in line.strip():
                        scn = next(f,'')
                        break
                    
            if 'Singlepulse_13C' in scn:
                drsexp.append(os.path.join(drsall[i], lsdrtr[j]))
                break
                
    return(drsexp)



####################################################################################################################

def plotFID2(foldPathNum, i, datAll):
    ACQ_Time = []
    with open(foldPathNum+'/method', 'r+') as f:
        for line in f:
            if '##$PVM_SpecAcquisitionTime=' in line.strip():
                ACQ_Time = float(line.strip()[27:])
                break


    co3 = datAll[foldPathNum]["co3"]

    fig, (ax1, ax2) = plt.subplots(2,1, dpi=174)

    # plt.figure(dpi=174)
    ax1.plot(np.linspace(0,ACQ_Time,len(np.real(co3[i-1]))), np.real(co3[i-1]), linewidth=0.5, label='real', color='purple')
    # ax1.set_label('real')

    ax2.plot(np.linspace(0,ACQ_Time,len(np.imag(co3[i-1]))), np.real(co3[i-1]), linewidth=0.5, label='imag', color='red')
    ax2.set_xlabel('time (ms)')
    ax1.legend()
    ax2.legend()
    # plt.xlabel('time (ms)')
    plt.savefig(foldPathNum+'/FID_NTP'+str(i)+'.png')
    plt.savefig(foldPathNum+'/FID_NTP'+str(i)+'.svg')
    plt.show()

def interFIDs(datAll, drsexp):    
    maxsc = np.max([len(datAll[drsexp[i]]["co3"]) for i in range(len(drsexp))])
    mm=interact(
        plotFID2,
        i = widgets.Select(options=[list(range(maxsc))[g]+1 for g in range(maxsc)], value = 1, description = "Scan:"), 
        foldPathNum = widgets.Dropdown(options=drsexp, value = drsexp[0], description = "Exp.:", layout=Layout(width='1000px')), 
        datAll = fixed(datAll),
        )
    
    
    
####################################################################################################################

def corPpm(drsexp, datAll):

    ini_All = {}
    ene_All = {}
    SNRs_All = {}
    
        
    for i in range(len(drsexp)):
        lnscn = len(datAll[drsexp[i]]["futmag"][0])

        mxdts = [np.max(datAll[drsexp[i]]["futmag"][k]) for k in range(len(datAll[drsexp[i]]["futmag"]))]
        nsdts = [np.std(datAll[drsexp[i]]["futmag"][k][0:round(lnscn*0.1)])  for k in range(len(datAll[drsexp[i]]["futmag"]))]

        thrs = np.mean(mxdts[len(mxdts)-round(len(mxdts)*0.2)::])   +   np.std(mxdts[len(mxdts)-round(len(mxdts)*0.2)::])*3
        dtins = []
        for k in range(len(mxdts)):
            if mxdts[k] > thrs:
                dtins.append(k)
                        
        ini_All[drsexp[i]] = dtins[0]
        ene_All[drsexp[i]] = dtins[-1::][0]
        SNRs_All[drsexp[i]] = [mxdts[k]/nsdts[k] for k in range(len(mxdts))]


        pyrin = np.where(datAll[drsexp[i]]["futmag"][ini_All[drsexp[i]]] == np.max(datAll[drsexp[i]]["futmag"][ini_All[drsexp[i]]]))[0][0] # Correct chemical shift
        corppm = 171 - datAll[drsexp[i]]["ppms"][pyrin]
        datAll[drsexp[i]]["ppms_cor"] = datAll[drsexp[i]]["ppms"]+corppm
        
    for i in range(len(drsexp)):
        datAll[drsexp[i]]['futmag_cor'] = []
        nse = np.mean(datAll[drsexp[i]]["futmag"][-1::][0])
        for k in range(len(datAll[drsexp[i]]["futmag"])):
            tmp = datAll[drsexp[i]]["futmag"][k] - nse
            datAll[drsexp[i]]['futmag_cor'].append(tmp)
        
    return(ini_All, ene_All, SNRs_All, datAll)


####################################################################################################################


def plotSingle2(foldPathNum, i, datAll2):
    
    ppms = datAll2[foldPathNum]["ppms_cor"]
    futmag = datAll2[foldPathNum]["futmag_cor"]
    
    plt.figure(dpi=170)

    # plt.plot([184, 184],[min(futmag[i]), max(futmag[i])])
    plt.plot(ppms, futmag[i], linewidth = 0.5)
    plt.gca().invert_xaxis()
    plt.ylim(min(futmag[i])-(max(futmag[i])*0.1), max(futmag[i])+(max(futmag[i])*0.1))
    plt.xlabel(r"$^{13}$C Chemical Shift (ppm)")
    
    plt.savefig(foldPathNum+'/SingleSpectra_NTP'+str(i)+'.png')
    plt.savefig(foldPathNum+'/SingleSpectra_NTP'+str(i)+'.svg')
    
    plt.show()
    
    
def interMags(datAll2, drsexp):    
    maxsc = np.max([len(datAll2[drsexp[i]]["futmag_cor"]) for i in range(len(drsexp))])
    mm=interact(
        plotSingle2,
        i = widgets.Select(options=[list(range(maxsc))[g]+1 for g in range(maxsc)], value = 5, description = "Scan:"), 
        foldPathNum = widgets.Dropdown(options=drsexp, value = drsexp[0], description = "Exp.:", layout=Layout(width='1000px')), 
        datAll2 = fixed(datAll2),
        )
    
    
    
    
####################################################################################################################

    
    
def plotSingle3(foldPathNum, datAll2):
    
    ppms = datAll2[foldPathNum]["ppms_cor"]
    sumup = datAll2[foldPathNum]["sumup"]
    
    plt.figure(dpi=170)

    # plt.plot([184, 184],[min(futmag[i]), max(futmag[i])])
    plt.plot(ppms, sumup, linewidth = 0.5)
    plt.gca().invert_xaxis()
    plt.ylim(min(sumup)-(max(sumup)*0.1), max(sumup)+(max(sumup)*0.1))
    plt.xlabel(r"$^{13}$C Chemical Shift (ppm)")
    plt.title("Sum Up")
    
    plt.savefig(foldPathNum+'/SumUP.png')
    plt.savefig(foldPathNum+'/SumUp.svg')
    
    plt.show()
    
    
# plotSingle2(drsexp[0], datAll2)

def interSumUp(datAll2, drsexp):    
    mm=interact(
        plotSingle3,
        foldPathNum = widgets.Dropdown(options=drsexp, value = drsexp[0], description = "Exp.:", layout=Layout(width='1000px')), 
        datAll2 = fixed(datAll2),
        )



####################################################################################################################


def plot3DStack(datAll2, drsexp, ini_All ,ene_All, exnall, exp, scns, col = "#80000aff", el=25, az=290, grad = False, sav=[], zoomlow = [-1, 0], zoomlowY = 0, lw = 1):
    
    dtt = datAll2[drsexp[exp]]['futmag_cor'][ini_All[drsexp[exp]]:ene_All[drsexp[exp]]+1][0:scns]
    ppms = datAll2[drsexp[exp]]["ppms_cor"]
    foldPath = drsexp[exp]
        
    mxl = [max(dtt[i]) for i in range(len(dtt))]
    mxm = max([max(dtt[i]) for i in range(len(dtt))])
    mxi = mxl.index(mxm)


    fig = plt.figure(figsize=(10,15))
    ax = plt.axes(projection='3d')
    xline = ppms[abs(zoomlow[1]):abs(zoomlow[0])]

    viridis = mpl.colormaps["viridis"].resampled(len(dtt))
    for i in range(len(dtt)):
        dtt_tmp = dtt[len(dtt)-1-i]
        if grad == False:
            ax.plot3D(xline, np.ones(len(xline))+(len(dtt)-1-i), dtt_tmp[abs(zoomlow[1]):abs(zoomlow[0])], color=col, linewidth=lw)
        else:
            ax.plot3D(xline, np.ones(len(xline))+(len(dtt)-1-i), dtt_tmp[abs(zoomlow[1]):abs(zoomlow[0])], color=viridis.colors[len(dtt)-1-i], linewidth=lw)
    ax.view_init(el, az)
    ax.invert_xaxis()

    ax.set_title(exnall[exp])
    ax.set_xlabel(r"$^{13}$C Chemical Shift (ppm)", fontsize=15, rotation=-10)
    ax.set_ylabel("Scan", fontsize=15, rotation=60)
    ax.set_zlabel("Intensity", fontsize=15, rotation=-10)




    mm = np.linspace(104, 150, 5100)
    mm1 = np.logspace(104, 104.999999, 5100)
    mm2 = 104 + ((mm1-min(mm1))*(104.999999 - 104))   /(max(mm1)-min(mm1))

    ts =(max(dtt[mxi])+((max(dtt[mxi])*0.05)-max(dtt[mxi])*zoomlowY/100))*0.05
    if zoomlowY <= 99:
        ax.set_zlim(min(dtt[mxi])-(ts), max(dtt[mxi])+((max(dtt[mxi])*0.05)-max(dtt[mxi])*zoomlowY/100))
    elif zoomlowY >= 99 and zoomlowY <= 104:
        ax.set_zlim(min(dtt[mxi])-(ts), max(dtt[mxi])+((max(dtt[mxi])*0.05)-max(dtt[mxi])*zoomlowY/100))
    else:
        a = (mm >= zoomlowY)
        ind = np.where(a == True)[0][0]
        ts =(max(dtt[mxi])+((max(dtt[mxi])*0.05)-max(dtt[mxi])*mm2[ind]/100))*0.05
        ax.set_zlim(min(dtt[mxi])-(abs(ts)), max(dtt[mxi])+((max(dtt[mxi])*0.05)-max(dtt[mxi])*mm2[ind]/100))
        


    if sav == True:
        plt.savefig(foldPath+'/Plots/Stack3D_Exp'+str(exp)+'.png')
        plt.savefig(foldPath+'/Plots/Stack3D_Exp'+str(exp)+'.svg')
        
        
        
def interStack3D(datAll2, drsexp, ini_All ,ene_All, exnall, grad=False):
    
    maxspc = max([len(datAll2[drsexp[m]]['futmag_cor'][ini_All[drsexp[m]]:ene_All[drsexp[m]]+1]) for m in range(len(drsexp))])
    maxdta = max([len(datAll2[drsexp[m]]['futmag_cor'][ini_All[drsexp[m]]:ene_All[drsexp[m]]+1][0]) for m in range(len(drsexp)) ])
    clls = ["#80000aff", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Black", "Grey"]

    mm=interact(
        plot3DStack,
        col = widgets.Dropdown(options=clls, value = "#80000aff", description = "Colour:"),
        grad = fixed(grad),
        az = widgets.FloatSlider(min=0,max=360,step=0.01,value=290, description ='azim', layout=Layout(width='1000px')),
        el = widgets.FloatSlider(min=0,max=100,step=0.01,value=25, description ='elev', layout=Layout(width='1000px')),
        exp = widgets.Dropdown(options=[list(range(len(drsexp)))[g] for g in range(len(drsexp))], value = 0, description = "Sub.Exp.:"),
        scns = widgets.IntSlider(min=1,max=maxspc,step=1,value=maxspc, description ='Scans:', layout=Layout(width='1000px')),
        sav = widgets.Checkbox(value = False, description = "Save Plot ON"),
        zoomlow=widgets.IntRangeSlider(min=-maxdta,max=0,step=1,value=[-maxdta, 0], readout = False, description ='Zoom X', layout=Layout(width='1000px')),
        zoomlowY=widgets.IntSlider(min=0,max=150,step=1,value=0, readout = False, description ='Zoom Y', layout=Layout(width='1000px')),
        lw = widgets.FloatSlider(min=0.0001,max=4,step=0.0001,value=1, readout = True, description ='LineWidth', layout=Layout(width='1000px')),
        drsexp=fixed(drsexp),
        datAll2=fixed(datAll2),
        ini_All = fixed(ini_All),
        ene_All = fixed(ene_All),
        exnall=fixed(exnall)) 
    
    
    
    
 ####################################################################################################################
   
    
    
    
    
    
def plotStack2D(datAll2, drsexp, ini_All ,ene_All, exnall, exp=0, scns=1, col = "#80000aff", multip = 1, zoomlowY=0, sav=False, zoomlow = [-1, 0]):
    
    dtt = datAll2[drsexp[exp]]['futmag_cor'][ini_All[drsexp[exp]]:ene_All[drsexp[exp]]+1][scns[0]:scns[1]]
    nrmfc_up = max([max(dtt[i]) for i in range(len(dtt))])
    ss = [max(dtt[i]) for i in range(len(dtt))].index(nrmfc_up)
    nrmfc_dw = min(dtt[ss])

    ppms = datAll2[drsexp[exp]]["ppms_cor"]
    foldPath = drsexp[exp]

    dtt_nrm = [(dtt[i]-nrmfc_dw)/(nrmfc_up-nrmfc_dw) for i in range(len(dtt))]

    tmprs = (100-zoomlowY)/100
    for i in range(len(dtt_nrm)):
        for j in range(len(dtt_nrm[i])):
            if dtt_nrm[i][j] > tmprs:
                dtt_nrm[i][j] = np.nan
            
                
    fig, ax = plt.subplots(figsize=(10, 4), dpi=170)
    for i in range(len(dtt)):
        ax.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], dtt_nrm[i][abs(zoomlow[1]):abs(zoomlow[0])]*multip+i+1, color=col, linewidth = 0.5)

    ax.set_ylabel("Scan")
    ax.set_xlabel(r"$^{13}$C Chemical Shift (ppm)")
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.set_title(exnall[exp])

    plt.gca().invert_xaxis()



    if sav == True:
        plt.savefig(foldPath+'/Plots/Stack2D_SubExp'+str(exp)+'.png')
        plt.savefig(foldPath+'/Plots/Stack2D_SubExp'+str(exp)+'.svg')

    plt.show()
    
    
def interStack2D(datAll2, drsexp, ini_All ,ene_All, exnall):
    
    maxspc = max([len(datAll2[drsexp[m]]['futmag_cor'][ini_All[drsexp[m]]:ene_All[drsexp[m]]+1]) for m in range(len(drsexp))])
    maxdta = max([len(datAll2[drsexp[m]]['futmag_cor'][ini_All[drsexp[m]]:ene_All[drsexp[m]]+1][0]) for m in range(len(drsexp)) ])
    clls = ["#80000aff", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Black", "Grey"]

    mm=interact(
        plotStack2D,
        col = widgets.Dropdown(options=clls, value = "#80000aff", description = "Colour:"),
        exp = widgets.Dropdown(options=[list(range(len(drsexp)))[g]+1 for g in range(len(drsexp))], value = 1, description = "Sub.Exp.:"),
        scns = widgets.IntRangeSlider(min=0,max=maxspc,step=1,value=[0, maxspc], description ='Scans:', layout=Layout(width='1000px')),
        sav = widgets.Checkbox(value = False, description = "Save Plot ON"),
        zoomlow=widgets.IntRangeSlider(min=-maxdta,max=0,step=1,value=[-maxdta, 0], readout = False, description ='Zoom X', layout=Layout(width='1000px')),
        multip=widgets.FloatSlider(min=1,max=200,step=0.01,value=0, readout = False, description ='Zoom Y', layout=Layout(width='1000px')),
        zoomlowY=widgets.FloatSlider(min=0,max=99.99,step=0.0001,value=0, readout = False, description ='Zoom Y2', layout=Layout(width='1000px')),
        drsexp=fixed(drsexp),
        datAll2=fixed(datAll2),
        ini_All = fixed(ini_All),
        ene_All = fixed(ene_All),
        exnall=fixed(exnall)) 
    
    
    


####################################################################################################################


def integRegs3(Lactate, Alanine, Piruvate, Bicarbonate, datAll2, drsexp, exnall, multip,ntp, zoomlow=[], exp = 0):
    futmag = datAll2[drsexp[exp]]['futmag_cor']
    alldat = np.reshape(futmag, np.prod(np.shape(futmag)))
    ppms = datAll2[drsexp[exp]]["ppms_cor"]
    foldPath = drsexp[exp]

    futmagnorm = [(futmag[i]-min(alldat))/(max(alldat)-min(alldat)) for i in range(len(futmag))]

    ts = 5

    fig, ax = plt.subplots(figsize=(12, 9))
    if bool(ntp[1]-ntp[0])==0 or (ntp[1]-ntp[0]) >= len(futmag):
        for i in range(len(futmag)):
            if bool(zoomlow)==0:
                plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
            else:
                plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmagnorm[i][abs(zoomlow[1]):abs(zoomlow[0])])*multip+i*ts, linewidth=0.5)
    else:
        for i in range(ntp[0], ntp[1]):
            if bool(zoomlow)==0:
                plt.plot(ppms, (futmagnorm[i])*multip+i*ts, linewidth=0.5)
            else:
                plt.plot(ppms[abs(zoomlow[1]):abs(zoomlow[0])], (futmagnorm[i][abs(zoomlow[1]):abs(zoomlow[0])])*multip+i*ts, linewidth=0.5)


    if bool(ntp[1]-ntp[0])==0:
        mv=len(futmag)
    else: 
        mv=ntp[1]
        
    ax.set_title(exnall[exp])
    
    # if os.path.isfile(foldPath+'/integRegTmp.npy'):
    #     min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate,min_Bicarbonate,max_Bicarbonate = np.load(foldPath+'/integRegTmp.npy')
    #     Piruvate = [max_Piruvate, min_Piruvate]
    #     Alanine = [max_Alanine, min_Alanine]
    #     Lactate = [max_Lactate, min_Lactate]
    #     Bicarbonate = [max_Bicarbonate, min_Bicarbonate]
    # else:
    #     pass
    
        
    ax.plot([abs(Piruvate[1]), abs(Piruvate[1])],[ntp[0]*ts, mv*ts], color = 'black')
    ax.plot([abs(Piruvate[0]), abs(Piruvate[0])],[ntp[0]*ts, mv*ts], color = 'black')

    ax.plot([abs(Alanine[1]), abs(Alanine[1])],[ntp[0]*ts, mv*ts], color = 'black')
    ax.plot([abs(Alanine[0]), abs(Alanine[0])],[ntp[0]*ts, mv*ts], color = 'black')

    ax.plot([abs(Lactate[1]), abs(Lactate[1])],[ntp[0]*ts, mv*ts], color = 'black')
    ax.plot([abs(Lactate[0]), abs(Lactate[0])],[ntp[0]*ts, mv*ts], color = 'black')

    ax.plot([abs(Bicarbonate[1]), abs(Bicarbonate[1])],[ntp[0]*ts, mv*ts], color = 'black')
    ax.plot([abs(Bicarbonate[0]), abs(Bicarbonate[0])],[ntp[0]*ts, mv*ts], color = 'black')


    ax.set_xlim(ax.get_xlim()[::-1])

    ax.fill_between(ppms, ntp[0]*ts, mv*ts, where= (ppms > abs(Piruvate[1])) & (ppms < abs(Piruvate[0])),
                    facecolor='red', alpha=0.2)

    ax.fill_between(ppms, ntp[0]*ts, mv*ts, where= (ppms > abs(Alanine[1])) & (ppms < abs(Alanine[0])),
                    facecolor='green', alpha=0.2)

    ax.fill_between(ppms, ntp[0]*ts, mv*ts, where= (ppms > abs(Lactate[1])) & (ppms < abs(Lactate[0])),
                    facecolor='yellow', alpha=0.2)

    ax.fill_between(ppms, ntp[0]*ts, mv*ts, where= (ppms > abs(Bicarbonate[1])) & (ppms < abs(Bicarbonate[0])),
                    facecolor='blue', alpha=0.2)
        
    ax.set_xlabel("ppm")
    plt.show()


    np.save(foldPath+'/integRegTmp.npy', [abs(Piruvate[1]),abs(Piruvate[0]),abs(Alanine[1]),abs(Alanine[0]),abs(Lactate[1]),abs(Lactate[0]),abs(Bicarbonate[1]),abs(Bicarbonate[0])])
    
    
def integralExtract(datAll2, drsexp, ini_All ,ene_All, exnall):    
    maxspc = max([len(datAll2[drsexp[m]]['futmag_cor'][ini_All[drsexp[m]]:ene_All[drsexp[m]]+1]) for m in range(len(drsexp))])
    maxdta = max([len(datAll2[drsexp[m]]['futmag_cor'][ini_All[drsexp[m]]:ene_All[drsexp[m]]+1][0]) for m in range(len(drsexp)) ])
        
    mm=interact(
            integRegs3,
            Piruvate = widgets.FloatRangeSlider(min=-185,max=-155,step=0.01,value=[-173, -168], readout = False, layout=Layout(width='1400px'), description ='Pyr'),
            Alanine = widgets.FloatRangeSlider(min=-185,max=-165,step=0.01,value=[-178,-175], readout = False, layout=Layout(width='1400px'), description ='Ala'),
            Lactate = widgets.FloatRangeSlider(min=-195,max=-170,step=0.01,value=[-184.5, -181.5], readout = False, layout=Layout(width='1400px'), description ='Lac'), 
            Bicarbonate = widgets.FloatRangeSlider(min=-175,max=-155,step=0.01,value=[-162, -160], readout = False, layout=Layout(width='1400px'), description ='BiC'), 
            zoomlow=widgets.IntRangeSlider(min=-maxdta,max=0,step=1,value=[-maxdta, 0], readout = False, description ='Zoom'),
            multip=widgets.FloatSlider(min=1,max=100,step=1,value=100), 
            exp = widgets.Dropdown(options=[list(range(len(drsexp)))[g] for g in range(len(drsexp))], value = 0, description = "Sub.Exp.:"),
            ntp=widgets.IntRangeSlider(min=0,max=maxspc,step=1,value=[np.min([ini_All[drsexp[i]] for i in range(len(drsexp))]), np.max([ene_All[drsexp[i]] for i in range(len(drsexp))])], layout=Layout(width='1000px')),
            drsexp=fixed(drsexp),
            datAll2=fixed(datAll2),
            ini_All = fixed(ini_All),
            ene_All = fixed(ene_All),
            exnall=fixed(exnall))
    
    
    
####################################################################################################################


def getIntsAll(drsexp, datAll2):

    IntsAll = {}
    intersAll = {}

    for exp in range(len(drsexp)):

        IntsAll[drsexp[exp]] = {}
        intersAll[drsexp[exp]] = {}

        futmag = datAll2[drsexp[exp]]['futmag_cor']
        sumup = datAll2[drsexp[exp]]['sumup']
        ppms = datAll2[drsexp[exp]]["ppms_cor"]
        foldPath = drsexp[exp]



        Piruvate = np.zeros(2)
        Alanine = np.zeros(2)
        Lactate = np.zeros(2)
        Bicarbonate = np.zeros(2)

        min_Piruvate,max_Piruvate,min_Alanine,max_Alanine,min_Lactate,max_Lactate,min_Bicarbonate,max_Bicarbonate = np.load(foldPath+'/integRegTmp.npy')
        Piruvate = [max_Piruvate, min_Piruvate]
        Alanine = [max_Alanine, min_Alanine]
        Lactate = [max_Lactate, min_Lactate]
        Bicarbonate = [max_Bicarbonate, min_Bicarbonate]


        IntsAll[drsexp[exp]]["Pyr"] = Piruvate
        IntsAll[drsexp[exp]]["Ala"] = Alanine
        IntsAll[drsexp[exp]]["Lac"] = Lactate
        IntsAll[drsexp[exp]]["Bic"] = Bicarbonate

        kss = ["Pyr","Ala","Lac","Bic"]

        for k in range(len(kss)):
            ppmSe1 = ppms[(ppms > IntsAll[drsexp[exp]][kss[k]][1]) & (ppms < IntsAll[drsexp[exp]][kss[k]][0])]
            intSe1 = sumup[(ppms > IntsAll[drsexp[exp]][kss[k]][1]) & (ppms < IntsAll[drsexp[exp]][kss[k]][0])]
            intersAll[drsexp[exp]][kss[k]] = auc(ppmSe1, intSe1)
            
            
        nms = ["Pyr","Ala","Lac","Bic"]
        dts = [intersAll[drsexp[exp]]["Pyr"], intersAll[drsexp[exp]]["Ala"], intersAll[drsexp[exp]]["Lac"], intersAll[drsexp[exp]]["Bic"]]
        df = pd.DataFrame([dts], columns = nms)

        df.to_csv(foldPath+"/SumUpIntegrals.csv", sep='\t')


    return(IntsAll, intersAll)



####################################################################################################################

def disSumUpInts(datAll2, IntsAll, intersAll, foldPathNum):
    sumup = datAll2[foldPathNum]['sumup']
    ppms = datAll2[foldPathNum]["ppms_cor"]
    foldPath = foldPathNum
        
    plt.figure(dpi=170)

    plt.plot(ppms, sumup, linewidth = 0.5)
    plt.gca().invert_xaxis()
    plt.ylim(min(sumup)-(max(sumup)*0.1), max(sumup)+(max(sumup)*0.1))
    plt.title(foldPathNum.split(os.sep)[-2])

    kss = ["Pyr","Ala","Lac","Bic"]
    clp = ["red", "green", "yellow", "blue"]


    for i in range(len(kss)):
        plt.plot([abs(IntsAll[foldPathNum][kss[i]][1]), abs(IntsAll[foldPathNum][kss[i]][1])],[0, max(sumup)], color = 'black')
        plt.plot([abs(IntsAll[foldPathNum][kss[i]][0]), abs(IntsAll[foldPathNum][kss[i]][0])],[0, max(sumup)], color = 'black')

        plt.fill_between(ppms, 0, max(sumup), where= (ppms > abs(IntsAll[foldPathNum][kss[i]][1])) & (ppms < abs(IntsAll[foldPathNum][kss[i]][0])),
                        facecolor=clp[i], alpha=0.2)
        
    plt.savefig(foldPath+'/PlotIntersSumUp.png')
    plt.savefig(foldPath+'/PlotIntersSumUp.svg')

    plt.show()
    
    print('Pyruvate Integral: '+str(intersAll[foldPathNum]['Pyr']))
    print('Alanine  Integral: '+str(intersAll[foldPathNum]['Ala']))
    print('Lactate  Integral: '+str(intersAll[foldPathNum]['Lac']))
    print('Bicarbonate  Integral: '+str(intersAll[foldPathNum]['Bic']))
    
    
def interSumUpInts(datAll2, drsexp, IntsAll, intersAll):    
    mm=interact(
        disSumUpInts,
        foldPathNum = widgets.Dropdown(options=drsexp, value = drsexp[0], description = "Exp.:", layout=Layout(width='1000px')), 
        datAll2 = fixed(datAll2),
        IntsAll = fixed(IntsAll), 
        intersAll = fixed(intersAll)
        )
    
    
####################################################################################################################

def getTempIntsAll(datAll2, drsexp, IntsAll):

    allDatsTemp = {}

    for exp in range(len(drsexp)):

        futmag = datAll2[drsexp[exp]]['futmag_cor']
        sumup = datAll2[drsexp[exp]]['sumup']
        ppms = datAll2[drsexp[exp]]["ppms_cor"]
        foldPath = drsexp[exp]
        allDatsTemp[drsexp[exp]] = {}


        kss = ["Pyr","Ala","Lac","Bic"]
        for j in range(len(kss)):
            ppmSe = ppms[(ppms > IntsAll[drsexp[exp]][kss[j]][1]) & (ppms < IntsAll[drsexp[exp]][kss[j]][0])]
            tmp = np.zeros(len(futmag))
            for k in range(len(futmag)):
                intSe = futmag[k][(ppms > IntsAll[drsexp[exp]][kss[j]][1]) & (ppms < IntsAll[drsexp[exp]][kss[j]][0])]
                tmp[k] = auc(ppmSe, intSe)
            allDatsTemp[drsexp[exp]][kss[j]] = tmp
            
        datT = np.transpose(pd.DataFrame([list(range(0, len(futmag)*5, 5)), allDatsTemp[drsexp[exp]]['Pyr'], allDatsTemp[drsexp[exp]]['Ala'], allDatsTemp[drsexp[exp]]['Lac'], allDatsTemp[drsexp[exp]]['Bic']]))
        datT2 = datT.rename(columns={0: "Time", 1: "Pyruvate", 2: "Alanine", 3: "Lactate", 4: "Bicarbonate"})
        datT2.to_csv(foldPath+"/TemporalDataAll.csv", index=False, header=True)
        
    return(allDatsTemp)

####################################################################################################################


def plotTimCrvs(allDatsTemp, foldPathAll, intersAll, shwP = True, shwA = True, shwL = True, shwB = True, sav = False):
    
    plt.figure(dpi=150)

    if shwP == True:
        plt.scatter(list(range(0,len(allDatsTemp[foldPathAll]["Pyr"])*5, 5)), allDatsTemp[foldPathAll]["Pyr"], label="Pyruvate")
    if shwL == True:
        plt.scatter(list(range(0,len(allDatsTemp[foldPathAll]["Lac"])*5, 5)), allDatsTemp[foldPathAll]["Lac"], label="Lactate")
    if shwA == True:
        plt.scatter(list(range(0,len(allDatsTemp[foldPathAll]["Ala"])*5, 5)), allDatsTemp[foldPathAll]["Ala"], label="Alanine")
    if shwB == True:
        plt.scatter(list(range(0,len(allDatsTemp[foldPathAll]["Bic"])*5, 5)), allDatsTemp[foldPathAll]["Bic"], label="Bicarbonate")



    plt.xlabel("Time (s)")
    leg = plt.legend(loc='upper right')
    if sav == True:
        plt.savefig(foldPathAll+'/PlotInters.png')
        plt.savefig(foldPathAll+'/PlotInters.svg')
        
    plt.show()

    kss = ["Pyr","Ala","Lac","Bic"]
    print('Ratio Pyruvate over Total Signal: '+str(intersAll[foldPathAll]['Pyr'] / np.sum([intersAll[foldPathAll][kss[j]] for j in range(len(kss))]) ))
    print('Ratio Lactate over Total Signal: '+str(intersAll[foldPathAll]['Lac'] / np.sum([intersAll[foldPathAll][kss[j]] for j in range(len(kss))]) ))
    print('Ratio Alanine over Total Signal: '+str(intersAll[foldPathAll]['Ala'] / np.sum([intersAll[foldPathAll][kss[j]] for j in range(len(kss))]) ))
    print('Ratio Bicarbonate over Total Signal: '+str(intersAll[foldPathAll]['Bic'] / np.sum([intersAll[foldPathAll][kss[j]] for j in range(len(kss))]) ))
    
    
    
    
def interTimssInts(drsexp, intersAll, allDatsTemp):    
    mm=interact(
        plotTimCrvs,
        foldPathAll = widgets.Dropdown(options=drsexp, value = drsexp[0], description = "Exp.:", layout=Layout(width='1000px')), 
        allDatsTemp = fixed(allDatsTemp),
        intersAll = fixed(intersAll),
        sav = widgets.Checkbox(value = False, description = "Save Plot ON"),
        shwP = widgets.Checkbox(value = True, description = "Show Pyr:"),
        shwA = widgets.Checkbox(value = True, description = "Show Ala:"),
        shwL = widgets.Checkbox(value = True, description = "Show Lac:"),
        shwB = widgets.Checkbox(value = True, description = "Show BiC:")
        )    
    
    
    
    
    
    
    
    
    
    
####################################################################################################################


def saveDatAll(drsexp, ini_All, ene_All, SNRs_All, intersAll, test, exnall,drsall,stsall,stiall,soiall):

    exp13C = [os.path.normpath(drsexp[i]).split(os.sep)[-1::][0] for i in range(len(drsexp))]
    inis = [ini_All[drsexp[i]] for i in range(len(drsexp))]
    enes = [ene_All[drsexp[i]] for i in range(len(drsexp))]
    snrs = [np.max(SNRs_All[drsexp[i]]) for i in range(len(drsexp))]

    pyrs = [intersAll[drsexp[i]]['Pyr'] for i in range(len(drsexp))]
    lacs = [intersAll[drsexp[i]]['Lac'] for i in range(len(drsexp))]
    alas = [intersAll[drsexp[i]]['Ala'] for i in range(len(drsexp))]
    bics = [intersAll[drsexp[i]]['Bic'] for i in range(len(drsexp))]

    df2 = pd.DataFrame(np.transpose([exnall,drsall,stsall,stiall,soiall,exp13C, inis, enes, snrs, pyrs, lacs, alas, bics ,
                                    pyrs/np.sum([pyrs,lacs,alas,bics]), lacs/np.sum([pyrs,lacs,alas,bics]), alas/np.sum([pyrs,lacs,alas,bics]), bics/np.sum([pyrs,lacs,alas,bics])]))

    df3 = df2.rename(columns={0: "ID", 1: "FilePath", 2: "StartScan", 3: "StartInject", 4: "StopInject", 5: "13C_Folder", 6: "ini", 7:"ene", 8:"SNR",
                        9: "Pyr", 10: "Lac", 11: "Ala", 12: "BiC",    13: "Pyr/Tot13C", 14: "Lac/Tot13C", 15: "Ala/Tot13C", 16: "BiC/Tot13C"})


    indx = [i for i in range(len(test)) if test[i].find('/')==0]
    foldPathS = test[0:indx[-1]]
    foldNumS = test[indx[-1]+1:]

    df3.to_csv(foldPathS+"/"+foldNumS[0:-5]+'_MagnitudeData.csv', index=False, header=True)
    df3.to_excel(foldPathS+"/"+foldNumS[0:-5]+'_MagnitudeData.xlsx', index=False, header=True)
    
    return(df3)