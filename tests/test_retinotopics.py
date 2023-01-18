import mousenet
import torch
import pandas as pd
import pdb
import pytest
import re
import numpy as np
from mousenet.mouse_cnn.data import Data

@pytest.fixture(scope="session")
def model(architecture="retinotopic", force=False):
    model = mousenet.load(architecture=architecture, pretraining=None, force=force)
    model.eval()
    return model

def test_retinotopics_runs(model):
    input = torch.rand(1, 3, 51, 79)
    results = model(input)
    print(f"results shape is {results.shape}")
    return True

# def test_retinotopics_loads_in_gpu(model):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     input = torch.rand(1, 3, 64, 64).to(device)
#     results = model(input)
#     print(results.shape)
def test_retinotopics_subfields(model):
    data = Data()
    input = torch.rand(1, 3, 51, 79)
    output = model.get_img_feature(input, ["VISpor5"], return_calc_graph=True, flatten=False)
    for area in output.keys():
        if area == "LGNd":
            continue
        x1, y1, x2, y2 = data.get_visual_field_shape(area)
        height, width = output[area].shape[-2:]
        assert width == (x2-x1)
        assert height == (y2-y1)
    

def test_num_channels(model):
    pdb.set_trace()
    data = Data()
    for layer in model.network.layers:
        source_name = layer.source_name
        if source_name == "input":
            continue
        
        layer_name = layer.source_name + layer.target_name    
        if source_name != "LGNd":
            area = re.split('[^[a-zA-Z]]*', layer.source_name)[0]
            depth = layer.source_name[len(area):]
            x1, y1, x2, y2 = data.get_visual_field_shape(f"{area}")
            n_source_channels = int(np.floor(data.get_num_neurons(area, depth) / ((y2-y1)*(x2-x1))))
            assert model.Convs[layer_name].in_channels == n_source_channels, f"n_source_channels mismatch for {layer_name}"
        
        area = re.split('[^[a-zA-Z]]*', layer.target_name)[0]
        depth = layer.target_name[len(area):]
        x1, y1, x2, y2 = data.get_visual_field_shape(f"{area}")
        n_target_channels = int(np.floor(data.get_num_neurons(area, depth) / ((y2-y1)*(x2-x1))))
        assert model.Convs[layer_name].out_channels == n_target_channels, f"n_target_channels mismatch for {layer_name}"
    

def test_kernel_sizes(model):
    # Visp2/3 -> Visp5. Stock is (5, 5)
    assert model.Convs["VISp2/3VISp5"].kernel_size == (5, 5)

    # Visli4 -> Vispor4. Stock is (17, 17)
    assert model.Convs["VISli4VISpor4"].kernel_size == (17, 17)
    
    # VISrl2/3 -> VISrl5. Stock is (7, 7)
    assert model.Convs["VISrl2/3VISrl5"].kernel_size == (7, 7)

    # VISal5 -> VISpor4. Stock is (1, 1)
    assert model.Convs["VISal5VISpor4"].kernel_size == (1, 1)
    
    #VISpor2/3 -> VISpor5. Stock is (1, 1)
    assert model.Convs["VISpor2/3VISpor5"].kernel_size == (1, 1)

def test_kernel_stride(model):
    #Visp2/3 -> Visp5. Stock is (1, 1)
    assert model.Convs["VISp2/3VISp5"].stride == (1, 1)

    #Visli4 -> Vispor4. Stock is (1, 1)
    assert model.Convs["VISli4VISpor4"].stride == (1, 1)
    
    #VISpor2/3 -> VISpor5. Stock is (1, 1)
    assert model.Convs["VISpor2/3VISpor5"].stride == (1, 1)

# TODO: Need to verify this once subfield is implemented
# out_sigma: ratio between output size and input size, 1/2 means reduce output size to 1/2 of the input size
# KmS = int((self.kernel_size-1/out_sigma))
# padding = int(KmS/2) or padding = (int(KmS/2), int(KmS/2+1), int(KmS/2), int(KmS/2+1))
def test_kernel_padding(model):
    #Visp2/3 -> Visp5. Stock is (1,1,1,1)
    assert model.Convs["VISp2/3VISp5"].mypadding.padding == (1,1,1,1)

    #Visli4 -> Vispor4. Stock is (8,8,8,8)
    assert model.Convs["VISli4VISpor4"].mypadding.padding == (6,7,6,7)
    
    #VISpor2/3 -> VISpor5. Stock is (1,1,1,1)
    assert model.Convs["VISpor2/3VISpor5"].mypadding.padding == (0,0,0,0)

def test_gaussian_height_and_width():
    params_df = extract_params(retinotopic=True, force=False)

    #Visp2/3 -> Visp5. Stock is 0.167338 and 1.8615418573925362
    assert np.isclose(params_df.iloc[9].gsh, 0.167338, rtol=1e-5)
    assert np.isclose(params_df.iloc[9].gsw , 1.8682405341867725, rtol=1e-5)

    #Visli4 -> Vispor4. Stock is 0.017436 and 8.978710
    assert np.isclose(params_df.iloc[21].gsh , 0.1237956, rtol=1e-6)
    assert np.isclose(params_df.iloc[21].gsw , 6.3748841, rtol=1e-6)
    
    #VISpor2/3 -> VISpor5. Stock is 0.167338 and 1.741991
    assert np.isclose(params_df.iloc[43].gsh , 0.07195534, rtol=1e-7)
    assert np.isclose(params_df.iloc[43].gsw , 0.74905613, rtol=1e-7)

def extract_params(retinotopic=False, force=False):
    """
    Get a dataframe of all parameters given whether the model should be retinotopic or not.
    Set force to true in order to regenerate mousenet, as opposed to using cache
    """
    net = mousenet.loader.generate_net(retinotopic, force)
    df_st = pd.DataFrame([(l.source_name, l.target_name) for l in net.layers])
    df_st.columns = ["source_name", "target_name"]
    
    df_params = pd.DataFrame([l.params.__dict__ for l in net.layers])
    
    df = pd.concat([df_st, df_params], axis=1)

    return df

if __name__ == "__main__":
    model = mousenet.load(architecture="retinotopic", pretraining=None, force=False)
    model.eval()
    input = torch.rand(1, 3, 100, 100)
    results = model(input)
    # test_num_channels(model)