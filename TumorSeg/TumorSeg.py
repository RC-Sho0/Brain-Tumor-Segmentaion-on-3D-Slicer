import logging
import os
import time

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import torch
import monai


from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Transpose, Activations, AsDiscrete
from monai import transforms

#
# TumorSeg
#

class TumorSeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Brain Tumor Segmentation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["HUFLIT THESIS"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Sho (RC.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#TumorSeg">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#



#
# TumorSegWidget
#

class TumorSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/TumorSeg.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ui.comboBox.addItems(['dynunet', 'segresnet', '3dfusion'])


        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)',self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        # self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        # self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        input 

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        # if self.parent.isEntered:
            # self.initializeParameterNode()
        pass


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def loadmodel(self, modelname):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## CHANGE YOUR TRAINED FOLDER PATH HERE
        trained_path = "/home/sowwn/SSD/mah_space/2023/slicer_extension/trained"

        model = torch.load(f'{trained_path}/{modelname.lower()}.pt')
        model.eval()
        return model

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        modelname = self.ui.comboBox.currentText
        print(f"Model: {modelname}")
        print(f"Run on {self.device}")

            # Compute results
        start = time.time()
        if modelname != 'None':
            # try:
                logging.info(f"Start at: {start}")
                model = self.loadmodel(modelname)
                seg_vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode","seg")
                pred = self.predict(model)
                pred = pred.permute(1,0,2)
                pred = pred.cpu().numpy().astype(np.uint8)

                logging.info(f"Predict shape: {pred.shape}")
                logging.info(f"Total time: {time.time()-start}")

                imageDirections = [[-1,0,0], [0,-1,0], [0,0,1]]
                imageOrigin = [0, 239, 0]
                seg_vol.SetIJKToRASDirections(imageDirections)
                seg_vol.SetOrigin(imageOrigin)
                slicer.util.updateVolumeFromArray(seg_vol, pred)
            # except:
            #     print("The model weight is not found!!!")
        print("=="*30)

    def merge(self, pred):
        pred = torch.squeeze(pred,0)
        c,w,h,d = pred.shape
        out = torch.zeros((5,w,h,d))
        out[1] = pred[0]*2
        out[2] = pred[1]*1
        out[4] = pred[2]*4
        out = torch.argmax(out,0)
        out =  Transpose((1,2,0))(out)
        return out


    def normalize(self, image):
            test_transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys="image", reader="NibabelReader"),
                    transforms.EnsureChannelFirstd(keys="image"),
                    transforms.EnsureTyped(keys="image"),
                    transforms.Orientationd(keys="image", axcodes="LPS"),
                    transforms.Spacingd(
                        keys="image",
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear"),
                    ),
                    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            )

            return test_transform(image)
 

    def predict(self, model):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        post_trans = Compose([Activations(sigmoid="True"), AsDiscrete(threshold=0.5)])
        inputs = self.setUp()
        with torch.no_grad():
            print(f"Paths: {inputs}")
            inputs = next(iter(self.normalize(inputs)))
            inputs = inputs['image'].to(device)
            inputs = torch.unsqueeze(inputs,0)
            
            val_outputs = sliding_window_inference(inputs=inputs,  roi_size=(128, 128, 128),  sw_batch_size=1,  predictor=model,  overlap=0.5,)
            val_outputs = post_trans(val_outputs)
            val_outputs = self.merge(val_outputs)
            
        return val_outputs

            
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        allnode = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        paths = {}
        # try:
        row = []
        for node in allnode:
            try:
                row.append(node.GetStorageNode().GetFullNameFromFileName())
            except: pass
        paths['image'] = row
        return [paths]
        # except: pass


    
#
# TumorSegTest
#

class TumorSegTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        allnode = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        data = torch.tensor(slicer.util.arrayFromVolume(allnode[0]))
        logging.info(f"Data shape: {data.shape}")
        seg_vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode","seg_test")
        data = data.numpy()
        slicer.util.updateVolumeFromArray(seg_vol, data)
        logging.info(f"Data shape: {data.shape}")
        

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()

        