# from coffea import hist
# from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
# from coffea import hist
import numpy as np
# from awkward import JaggedArray
from pprint import pprint as pprint
# from coffea.util import load, save
# import awkward
import config
# import uproot
# from pprint import pprint as pprint
# import sys


from numpy import random
from scipy.spatial.distance import pdist, cdist
import ROOT
ROOT.gROOT.SetBatch(True)


def energy(x, y, method='log'):
    x = np.atleast_2d(x).T
    y = np.atleast_2d(y).T
#     print(x.shape)
#     print(y.shape)
#     dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    dx, dxy = pdist(x), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        epsilon = 1e-10
#         dx, dy, dxy = -np.log(dx+epsilon), -np.log(dy+epsilon), -np.log(dxy+epsilon)
        dx, dxy = -np.log(dx+epsilon), -np.log(dxy+epsilon)
    elif method == 'gaussian':
        s = 1
#         dx, dy, dxy = np.exp(-(dx**2)/(2*s**2)), np.exp(-(dy**2)/(2*s**2)), np.exp(-(dxy**2)/(2*s**2))
        dx, dxy = np.exp(-(dx**2)/(2*s**2)), np.exp(-(dxy**2)/(2*s**2))
    elif method == 'linear':
        pass
    else:
        raise ValueError
#     return (-dxy.sum() / (n * m) + dx.sum() / n**2 + dy.sum() / m**2)
    return (-dxy.sum() / (n * m) + dx.sum() / n**2 )


# Look at ProcessorABC to see the expected methods and what they are supposed to do
class FitterProcessor(processor.ProcessorABC):
  def __init__(self,):
    self._accumulator = processor.dict_accumulator({
        # # cutflow
        # 'cutflow': processor.defaultdict_accumulator(float),
        # # histograms - No Kinematical cuts
        # 'h_leadmu_pt_noKinCuts': hist.Hist("Counts", dataset_axis, leadmu_pt_axis_noKinCuts),
        })

  @property
  def accumulator(self):
    return self._accumulator

  def process(self, ds):
    ####################################
    # get accumulator and load data
    ####################################
    output = self.accumulator.identity()
    dataset = ds['dataset_name']
    year = ds['year']

    ########################################
    # done
    ########################################
    c1 = ROOT.TCanvas()


    x = ROOT.RooRealVar("x","x",-10,10)
    p0_ = ROOT.RooRealVar("p0_","v",0.5,0,10)
    p1_ = ROOT.RooRealVar("p1_","v",1,0,10)
    p2_ = ROOT.RooRealVar("p2_","v",0.5,0,10)
    p3_ = ROOT.RooRealVar("p3_","v",1,0,10)
    p4_ = ROOT.RooRealVar("p4_","v",0.5,0,10)
     
        
    # mean = ROOT.RooRealVar("mean","Mean of Gaussian",-10,10)
    # sigma = ROOT.RooRealVar("sigma","Width of Gaussian",3,-10,10)

    true_model = ROOT.RooBernstein("true_model","RooBernstein",x,ROOT.RooArgList(p0_,p1_,p2_,p3_))
    # true_model = ROOT.RooGaussian("gauss","gauss(x,mean,sigma)",x,mean,sigma)


    # mean = ROOT.RooRealVar("mean","mean",0,-10,10)
    # sigma = ROOT.RooRealVar("sigma","sigma",2,0.,10)
    # sig = ROOT.RooGaussian("sig","signal p.d.f.",x,mean,sigma)

    # coef0 = ROOT.RooRealVar("c0","coefficient #0",1.0,-1.,1)
    # coef1 = ROOT.RooRealVar("c1","coefficient #1",0.1,-1.,1)
    # coef2 = ROOT.RooRealVar("c2","coefficient #2",-0.1,-1.,1)
    # bkg = ROOT.RooChebychev("bkg","background p.d.f.",x,ROOT.RooArgList(coef0,coef1,coef2))

    # fsig = ROOT.RooRealVar("fsig","signal fraction",0.1,0.,1.)

    # # model(x) = fsig*sig(x) + (1-fsig)*bkg(x)
    # true_model = ROOT.RooAddPdf("model","model",ROOT.RooArgList(sig,bkg),ROOT.RooArgList(fsig))


    ################################################################
    ################################################################
    ################################################################


    p0 = ROOT.RooRealVar("p0","v",1,0,10)
    p1 = ROOT.RooRealVar("p1","v",1,0,10)
    p2 = ROOT.RooRealVar("p2","v",1,0,100)
    p3 = ROOT.RooRealVar("p3","v",1,0,10)
    p4 = ROOT.RooRealVar("p4","v",1,0,10)
    p5 = ROOT.RooRealVar("p5","v",1,0,10)


    n_data = 5000
    data = true_model.generate(ROOT.RooArgSet(x),n_data)

    xframe = x.frame()
    data.plotOn(xframe, ROOT.RooLinkedList())
    true_model.plotOn(xframe,ROOT.RooFit.LineColor(ROOT.kGray))
    # data_array = roodataset2numpy(data, n_data)


    # leg1 = ROOT.TLegend(0.65,0.73,0.86,0.87);
    leg1 = ROOT.TLegend();
    # leg1.SetFillColor(ROOT.kWhite);
    # leg1.SetLineColor(ROOT.kWhite);
    leg1.AddEntry(data,"Data", "P");
    # leg1->AddEntry(“model”,“Signal + background”,“LP”);
    # leg1->AddEntry(“background”,“Background only”, “LP”);
    # leg1->AddEntry(“signal only”,“Signal only”, “LP”);

    print("##################################")
    print("--> 0th order:")
    p0.setVal(1)
    p1.setVal(1)
    p2.setVal(1)
    p3.setVal(1)
    p4.setVal(1)
    p5.setVal(1)
    pol_0_order = ROOT.RooBernstein("pol_0_order","RooBernstein",x,ROOT.RooArgList(p0))
    pol_0_order_fit = pol_0_order.fitTo(data,ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save())
    nll_0 = pol_0_order_fit.minNll()
    p0.setConstant(True)
    p1.setConstant(True)
    p2.setConstant(True)
    p3.setConstant(True)
    p4.setConstant(True)
    p5.setConstant(True)
    # en_0 = get_energy(data_array, pol_0_order, n_data)
    # plt.hist(en_0[1], 30) ;
    # plt.axvline(x=en_0[0], color="tab:red")
    # plt.xlabel('energy')
    # plt.ylabel('Probability')
    # plt.title('0-order ; p-value: '+str(en_0[2]))
    # plt.show()
    pol_0_order.plotOn(xframe,ROOT.RooFit.LineColor(2+0), ROOT.RooFit.Name("pol_0_order"))
    leg1.AddEntry(xframe.findObject("pol_0_order"),"0th order", "L");

    print("p0: "+str(p0.getVal()))
    p0.setConstant(False)
    p1.setConstant(False)
    p2.setConstant(False)
    p3.setConstant(False)
    p4.setConstant(False)
    p5.setConstant(False)

    print("##################################")
    print("--> 1st order:")
    p0.setVal(1)
    p1.setVal(1)
    p2.setVal(1)
    p3.setVal(1)
    p4.setVal(1)
    p5.setVal(1)
    pol_1_order = ROOT.RooBernstein("pol_1_order","RooBernstein",x,ROOT.RooArgList(p0,p1))
    pol_1_order_fit = pol_1_order.fitTo(data,ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save())
    nll_1 = pol_1_order_fit.minNll()
    p0.setConstant(True)
    p1.setConstant(True)
    p2.setConstant(True)
    p3.setConstant(True)
    p4.setConstant(True)
    p5.setConstant(True)
    # en_1 = get_energy(data_array, pol_1_order, n_data)
    # plt.hist(en_1[1], 30) ;
    # plt.axvline(x=en_1[0], color="tab:red")
    # plt.xlabel('energy')
    # plt.ylabel('Probability')
    # plt.title('1-order ; p-value: '+str(en_1[2]))
    # plt.show()
    pol_1_order.plotOn(xframe,ROOT.RooFit.LineColor(2+1), ROOT.RooFit.Name("pol_1_order"))
    leg1.AddEntry(xframe.findObject("pol_1_order"),"1st order", "L");

    print("p0: "+str(p0.getVal()))
    print("p1: "+str(p1.getVal()))
    p0.setConstant(False)
    p1.setConstant(False)
    p2.setConstant(False)
    p3.setConstant(False)
    p4.setConstant(False)
    p5.setConstant(False)

    print("##################################")
    print("--> 2nd order:")
    p0.setVal(1)
    p1.setVal(1)
    p2.setVal(1)
    p3.setVal(1)
    p4.setVal(1)
    p5.setVal(1)
    pol_2_order = ROOT.RooBernstein("pol_2_order","RooBernstein",x,ROOT.RooArgList(p0,p1,p2))
    pol_2_order_fit = pol_2_order.fitTo(data,ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save())
    nll_2 = pol_2_order_fit.minNll()
    p0.setConstant(True)
    p1.setConstant(True)
    p2.setConstant(True)
    p3.setConstant(True)
    p4.setConstant(True)
    p5.setConstant(True)
    # en_2 = get_energy(data_array, pol_2_order, n_data)
    # plt.hist(en_2[1], 30) ;
    # plt.axvline(x=en_2[0], color="tab:red")
    # plt.xlabel('energy')
    # plt.ylabel('Probability')
    # plt.title('2-order ; p-value: '+str(en_2[2]))
    # plt.show()
    pol_2_order.plotOn(xframe,ROOT.RooFit.LineColor(2+2), ROOT.RooFit.Name("pol_2_order"))
    leg1.AddEntry(xframe.findObject("pol_2_order"),"2nd order", "L");

    print("p0: "+str(p0.getVal()))
    print("p1: "+str(p1.getVal()))
    print("p2: "+str(p2.getVal()))
    p0.setConstant(False)
    p1.setConstant(False)
    p2.setConstant(False)
    p3.setConstant(False)
    p4.setConstant(False)
    p5.setConstant(False)

    print("##################################")
    print("--> 3rd order:")
    p0.setVal(1)
    p1.setVal(1)
    p2.setVal(1)
    p3.setVal(1)
    p4.setVal(1)
    p5.setVal(1)
    pol_3_order = ROOT.RooBernstein("pol_3_order","RooBernstein",x,ROOT.RooArgList(p0,p1,p2,p3))
    pol_3_order_fit = pol_3_order.fitTo(data,ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save())
    nll_3 = pol_3_order_fit.minNll()
    p0.setConstant(True)
    p1.setConstant(True)
    p2.setConstant(True)
    p3.setConstant(True)
    p4.setConstant(True)
    p5.setConstant(True)
    # en_3 = get_energy(data_array, pol_3_order, n_data)
    # plt.hist(en_3[1], 30) ;
    # plt.axvline(x=en_3[0], color="tab:red")
    # plt.xlabel('energy')
    # plt.ylabel('Probability')
    # plt.title('3-order ; p-value: '+str(en_3[2]))
    # plt.show()

    pol_3_order.plotOn(xframe,ROOT.RooFit.LineColor(2+3), ROOT.RooFit.Name("pol_3_order"))
    leg1.AddEntry(xframe.findObject("pol_3_order"),"3rd order", "L");

    print("p0: "+str(p0.getVal()))
    print("p1: "+str(p1.getVal()))
    print("p2: "+str(p2.getVal()))
    print("p3: "+str(p3.getVal()))
    p0.setConstant(False)
    p1.setConstant(False)
    p2.setConstant(False)
    p3.setConstant(False)
    p4.setConstant(False)
    p5.setConstant(False)

    print("##################################")
    print("--> 4th order:")
    p0.setVal(1)
    p1.setVal(1)
    p2.setVal(1)
    p3.setVal(1)
    p4.setVal(1)
    p5.setVal(1)
    pol_4_order = ROOT.RooBernstein("pol_4_order","RooBernstein",x,ROOT.RooArgList(p0,p1,p2,p3,p4))
    pol_4_order_fit = pol_4_order.fitTo(data,ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save())
    nll_4 = pol_4_order_fit.minNll()
    p0.setConstant(True)
    p1.setConstant(True)
    p2.setConstant(True)
    p3.setConstant(True)
    p4.setConstant(True)
    p5.setConstant(True)
    # en_4 = get_energy(data_array, pol_4_order, n_data)
    # plt.hist(en_4[1], 30) ;
    # plt.axvline(x=en_4[0], color="tab:red")
    # plt.xlabel('energy')
    # plt.ylabel('Probability')
    # plt.title('4-order ; p-value: '+str(en_4[2]))
    # plt.show()
    pol_4_order.plotOn(xframe,ROOT.RooFit.LineColor(2+4), ROOT.RooFit.Name("pol_4_order"))
    leg1.AddEntry(xframe.findObject("pol_4_order"),"4th order", "L");

    print("p0: "+str(p0.getVal()))
    print("p1: "+str(p1.getVal()))
    print("p2: "+str(p2.getVal()))
    print("p3: "+str(p3.getVal()))
    print("p4: "+str(p4.getVal()))
    p0.setConstant(False)
    p1.setConstant(False)
    p2.setConstant(False)
    p3.setConstant(False)
    p4.setConstant(False)
    p5.setConstant(False)

    print("##################################")
    print("--> 5th order:")
    p0.setVal(1)
    p1.setVal(1)
    p2.setVal(1)
    p3.setVal(1)
    p4.setVal(1)
    p5.setVal(1)
    pol_5_order = ROOT.RooBernstein("pol_5_order","RooBernstein",x,ROOT.RooArgList(p0,p1,p2,p3,p4,p5))
    pol_5_order_fit = pol_5_order.fitTo(data,ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save())
    nll_5 = pol_5_order_fit.minNll()
    p0.setConstant(True)
    p1.setConstant(True)
    p2.setConstant(True)
    p3.setConstant(True)
    p4.setConstant(True)
    p5.setConstant(True)
    # en_5 = get_energy(data_array, pol_5_order, n_data)
    # plt.hist(en_5[1], 30) ;
    # plt.axvline(x=en_5[0], color="tab:red")
    # plt.xlabel('energy')
    # plt.ylabel('Probability')
    # plt.title('5-order ; p-value: '+str(en_5[2]))
    # plt.show()
    pol_5_order.plotOn(xframe,ROOT.RooFit.LineColor(2+5), ROOT.RooFit.Name("pol_5_order"))
    leg1.AddEntry(xframe.findObject("pol_5_order"),"5th order", "L");

    print("p0: "+str(p0.getVal()))
    print("p1: "+str(p1.getVal()))
    print("p2: "+str(p2.getVal()))
    print("p3: "+str(p3.getVal()))
    print("p4: "+str(p4.getVal()))
    print("p5: "+str(p5.getVal()))
    p0.setConstant(False)
    p1.setConstant(False)
    p2.setConstant(False)
    p3.setConstant(False)
    p4.setConstant(False)
    p5.setConstant(False)


    # data.plotOn(xframe, ROOT.RooLinkedList())
    xframe.Draw()
    leg1.Draw()
    c1.Draw()

    # Save Plot
    filename = "plots/"+year+"/mass_" + ds["dataset_name"] + "_"+ ds["year"] 
    c1.SaveAs(filename+".png")
    c1.SaveAs(filename+".pdf")


    ########################################
    # done
    ########################################
    # return output - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })

  def postprocess(self, accumulator):
    # return accumulator - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })



