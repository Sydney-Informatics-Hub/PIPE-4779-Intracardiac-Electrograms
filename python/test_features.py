# test FeatureExtraction

from features import FeatureExtraction

def test_feature_extraction():
    target = 'scar'
    wavefront = 'SR'
    outpath = 'test
    inpath = '../../../data/generated'
    fname_csv = 'NestedDataS18.csv'
    fe = FeatureExtraction(inpath, fname_csv, outpath)
    fe.run_wavefront_target(wavefront, target)
    assert fe.selected_features.shape[0] > 0
    assert fe.selected_features.shape[1] > 0