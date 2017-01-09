from sound_field_analysis import sph
from py.test import approx


def test_sph_harm():
    assert sph.sph_harm(0, 0, 0, 0, type='complex') == approx(0.282094791773878)
    assert sph.sph_harm(0, 1, 0.1, 0.1, type='complex') == approx(0.486161534508712)
    assert sph.sph_harm(-2, 2, -0.1, -0.1, type='complex') == approx(0.003773142018527 + 0.000764853752555j)
    assert sph.sph_harm(2, 2, -0.1, 0.1, type='complex') == approx(0.003773142018527 - 0.000764853752555j)
    assert sph.sph_harm(17, 17, 7, 6, type='complex') == approx(2.202722537106286e-10 - 8.811259608537731e-11j)
    assert sph.sph_harm(-17, 17, 6, -7, type='complex') == approx(4.945682459644879e-05 - 4.843297071701738e-04j)
