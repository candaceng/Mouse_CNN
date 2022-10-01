import mousenet
import torch
import pandas as pd
import pdb
import pytest
import numpy as np

@pytest.fixture(scope="session")
def model(architecture="retinotopic", force=False):
    model = mousenet.load(architecture=architecture, pretraining=None, force=force)
    model.eval()
    return model

def test_retinotopics_runs(model):
    input = torch.rand(1, 3, 64, 64)
    results = model(input)
    print(f"results shape is {results.shape}")
    return True

# def test_retinotopics_loads_in_gpu(model):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     input = torch.rand(1, 3, 64, 64).to(device)
#     results = model(input)
#     print(results.shape)

def test_num_channels(model):
    pdb.set_trace()
    #Visp2/3 -> Visp5. Stock is 42
    assert model.Convs["VISp2/3VISp5"].in_channels == 42

    #Visli4 -> Vispor4. Stock is 5
    assert model.Convs["VISli4VISpor4"].in_channels == 8
    
    #VISpor2/3 -> VISpor5. Stock 29
    assert model.Convs["VISpor2/3VISpor5"].in_channels == 159

def test_kernel_sizes(model):
    # Visp2/3 -> Visp5. Stock is (3, 3)
    assert model.Convs["VISp2/3VISp5"].kernel_size == (3, 3)

    # Visli4 -> Vispor4. Stock is (15, 15)
    assert model.Convs["VISli4VISpor4"].kernel_size == (15, 15)
    
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
    test_num_channels()