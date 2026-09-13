"""Native parameter domains and embedded sources for autonomous birdsong Faust programs.

The Faust source strings embed verbatim host parameter labels (e.g. RA inh, wall-loss alph) that
are load-bearing Faust UI addresses and cannot be spell-corrected.
"""

from collections.abc import Mapping
from types import MappingProxyType

from synth_setter.data.vst.param_spec import ContinuousParameter
from synth_setter.param_spec_name import ParamSpecName

type BirdsongParameterDomain = tuple[str, float, float]

BIRDSONG_PARAMETER_DOMAINS: Mapping[
    ParamSpecName, tuple[BirdsongParameterDomain, ...]
] = MappingProxyType(
    {
        ParamSpecName('faust_single_syrinx'): (
            ('/single_syrinx/0_Air/Air_density', 1.0, 1.4),
            ('/single_syrinx/0_Air/Speed_of_sound', 300.0, 380.0),
            ('/single_syrinx/1_Anatomy/Bronchus_radius_a', 0.5, 10.0),
            ('/single_syrinx/1_Anatomy/Membrane_half-width_h', 0.5, 10.0),
            ('/single_syrinx/1_Anatomy/Membrane_thickness_d', 10.0, 500.0),
            ('/single_syrinx/1_Anatomy/Membrane_density', 500.0, 1500.0),
            ('/single_syrinx/1_Anatomy/Trachea_length_L', 5.0, 120.0),
            ('/single_syrinx/1_Anatomy/Mouth_horn_radius_b', 1.0, 20.0),
            ('/single_syrinx/1_Anatomy/Bronchial_volume_V', 0.05, 5.0),
            ('/single_syrinx/1_Anatomy/Air-sac_impedance_ZG', 0.1, 20.0),
            ('/single_syrinx/2_Valve/Coupling_eps_mode_2', 0.0, 1.0),
            ('/single_syrinx/2_Valve/Inertance_scale__0_=_quasi-static_', 0.0, 3.0),
            ('/single_syrinx/2_Valve/Suction_scale', 0.0, 3.0),
            ('/single_syrinx/2_Valve/Mode_2_ratio_f2_f1', 1.0, 4.0),
            ('/single_syrinx/2_Valve/Damping_kappa', 10.0, 5000.0),
            ('/single_syrinx/2_Valve/Contact_damping_E', 1.0, 100.0),
            ('/single_syrinx/2_Valve/Tissue_mass_eta', 0.0, 50.0),
            ('/single_syrinx/2_Valve/Phonation_rest_opening_x0', -2.0, 5.0),
            ('/single_syrinx/2_Valve/Force_constant_A1', 0.3, 3.0),
            ('/single_syrinx/2_Valve/Mass_constant_A3', 0.3, 3.0),
            ('/single_syrinx/2_Valve/Coupling_eps_mode_1', 0.0, 1.0),
            ('/single_syrinx/3_Tract/Wall_loss_coefficient', 0.0, 10.0),
            ('/single_syrinx/3_Tract/Loss_evaluated_at', 100.0, 8000.0),
            ('/single_syrinx/3_Tract/Beak_lowpass', 500.0, 20000.0),
            ('/single_syrinx/4_Output/Master_gain', -60.0, 0.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_exc_bias', -8.4, 1.6),
            ('/single_syrinx/A_Syllable/1_RA/RA_exc_from_HVC_burst', 0.0, 10.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_exc_self', 1.0, 11.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_exc_from_RA_inh', -8.0, 2.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_inh_bias', -12.0, -2.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_inh_from_HVC_stop_burst', -5.0, 5.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_inh_from_RA_exc', 1.0, 11.0),
            ('/single_syrinx/A_Syllable/1_RA/RA_inh_self', -2.0, 8.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_exc_from_RA', 5.0, 15.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_exc_from_IA_pulse', -4.0, 6.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_exc_self', 5.0, 15.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_exc_from_ER_inh', -6.1, 3.9),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_inh_bias', -16.5, -6.5),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_inh_from_RA', -5.0, 5.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_inh_from_ER_exc', 5.0, 15.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_inh_self', -3.0, 7.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/IR_bias', -5.0, 5.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/IR_from_RA', 5.0, 15.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/IR_from_ER_exc', -15.0, -5.0),
            ('/single_syrinx/A_Syllable/2_Respiratory/ER_exc_bias', -12.45, -2.45),
            ('/single_syrinx/A_Syllable/3_nXII/vs_bias', -8.0, 2.0),
            ('/single_syrinx/A_Syllable/3_nXII/vs_from_IA_pulse', -4.0, 6.0),
            ('/single_syrinx/A_Syllable/3_nXII/vs_from_RA', -4.5, 5.5),
            ('/single_syrinx/A_Syllable/3_nXII/vs_from_ER', -4.5, 5.5),
            ('/single_syrinx/A_Syllable/3_nXII/vs_from_IR', -5.0, 5.0),
            ('/single_syrinx/A_Syllable/3_nXII/vtb_bias', -8.0, 2.0),
            ('/single_syrinx/A_Syllable/3_nXII/vtb_from_IR', 5.0, 15.0),
            ('/single_syrinx/A_Syllable/4_Scaling/Pressure_gain', 0.666667, 6.0),
            ('/single_syrinx/A_Syllable/4_Scaling/Pressure_offset', -2.0, 2.0),
            ('/single_syrinx/A_Syllable/4_Scaling/Tension_L_gain', 9.66667, 87.0),
            ('/single_syrinx/A_Syllable/4_Scaling/Tension_L_offset', -10.0, 10.0),
            ('/single_syrinx/A_Syllable/4_Scaling/Gating_L_vtb_gain', 13.3333, 120.0),
            ('/single_syrinx/A_Syllable/5_Timing/Syllable_rate', 0.95, 8.55),
            ('/single_syrinx/A_Syllable/5_Timing/IA_pulse_width', 6.66667, 60.0),
            ('/single_syrinx/A_Syllable/5_Timing/IA_to_nXII_delay', 3.33333, 30.0),
            ('/single_syrinx/A_Syllable/5_Timing/IA_to_HVC_to_RA_delay', 10.0, 90.0),
            ('/single_syrinx/A_Syllable/5_Timing/HVC_burst_width', 10.0, 90.0),
            ('/single_syrinx/A_Syllable/5_Timing/HVC_stop_burst_delay', 100.0, 900.0),
            ('/single_syrinx/A_Syllable/5_Timing/HVC_periodic_amount', 0.0, 10.0),
            ('/single_syrinx/A_Syllable/5_Timing/HVC_periodic_rate', 8.33333, 75.0),
            ('/single_syrinx/A_Syllable/6_Bilateral/dtb_L_bias', -8.0, 2.0),
            ('/single_syrinx/A_Syllable/6_Bilateral/dtb_L_from_IA_pulse', 5.0, 15.0),
            ('/single_syrinx/A_Syllable/6_Bilateral/dtb_L_from_RA', -5.0, 5.0),
            ('/single_syrinx/A_Syllable/6_Bilateral/Gating_L_dtb_gain', 13.3333, 120.0),
            ('/single_syrinx/A_Syllable/6_Bilateral/Gating_L_offset', -10.0, 10.0),
            ('/single_syrinx/B_Mapping/Pressure_scale', 333.333, 3000.0),
            ('/single_syrinx/B_Mapping/Tension_reference_frequency', 500.0, 4500.0),
            ('/single_syrinx/B_Mapping/Tension_reference_value', 9.66667, 87.0),
            ('/single_syrinx/B_Mapping/Gating_to_opening', 0.0333333, 0.3),
        ),
        ParamSpecName('faust_bilateral_syrinx'): (
            ('/bilateral_syrinx/0_Air/Air_density', 1.0, 1.4),
            ('/bilateral_syrinx/0_Air/Speed_of_sound', 300.0, 386.0),
            ('/bilateral_syrinx/1_Anatomy_L/Bronchus_radius', 0.833333, 7.5),
            ('/bilateral_syrinx/1_Anatomy_L/Bronchus_length', 4.66667, 42.0),
            ('/bilateral_syrinx/1_Anatomy_L/Membrane_half-width', 0.583333, 5.25),
            ('/bilateral_syrinx/1_Anatomy_L/Lower_bronchus_volume', 0.09, 0.81),
            ('/bilateral_syrinx/1_Anatomy_L/Air-sac_impedance', 0.666667, 6.0),
            ('/bilateral_syrinx/1_Anatomy_L/Membrane_thickness', 33.3333, 300.0),
            ('/bilateral_syrinx/1_Anatomy_L/Membrane_density', 500.0, 2000.0),
            ('/bilateral_syrinx/2_Anatomy_R/Bronchus_radius', 0.833333, 7.5),
            ('/bilateral_syrinx/2_Anatomy_R/Bronchus_length', 4.66667, 42.0),
            ('/bilateral_syrinx/2_Anatomy_R/Membrane_half-width', 0.583333, 5.25),
            ('/bilateral_syrinx/2_Anatomy_R/Lower_bronchus_volume', 0.09, 0.81),
            ('/bilateral_syrinx/2_Anatomy_R/Air-sac_impedance', 0.666667, 6.0),
            ('/bilateral_syrinx/2_Anatomy_R/Membrane_thickness', 33.3333, 300.0),
            ('/bilateral_syrinx/2_Anatomy_R/Membrane_density', 500.0, 2000.0),
            ('/bilateral_syrinx/3_Anatomy_shared/Trachea_length', 7.66667, 69.0),
            ('/bilateral_syrinx/3_Anatomy_shared/Trachea_radius', 1.16667, 10.5),
            ('/bilateral_syrinx/3_Anatomy_shared/Mouth_horn_radius', 3.33333, 30.0),
            ('/bilateral_syrinx/4_Valves/Mass_constant_A3', 0.333333, 3.0),
            ('/bilateral_syrinx/4_Valves/Coupling_eps_mode_1', 0.5, 1.5),
            ('/bilateral_syrinx/4_Valves/Coupling_eps_mode_2', 0.5, 1.5),
            ('/bilateral_syrinx/4_Valves/Inertance_scale', 0.333333, 3.0),
            ('/bilateral_syrinx/4_Valves/Suction_scale', 0.333333, 3.0),
            ('/bilateral_syrinx/4_Valves/Phonation_rest_opening_L', -1.5, 1.5),
            ('/bilateral_syrinx/4_Valves/Phonation_rest_opening_R', -1.5, 1.5),
            ('/bilateral_syrinx/4_Valves/Mode_2_ratio_f2_f1', 1.2, 2.0),
            ('/bilateral_syrinx/4_Valves/Damping_kappa', 523.667, 4713.0),
            ('/bilateral_syrinx/4_Valves/Contact_damping_E', 10.0, 100.0),
            ('/bilateral_syrinx/4_Valves/Tissue_mass_eta', 3.33333, 30.0),
            ('/bilateral_syrinx/4_Valves/Force_constant_A1', 0.333333, 3.0),
            ('/bilateral_syrinx/5_Tract/Wall_loss_coefficient', 0.666667, 6.0),
            ('/bilateral_syrinx/5_Tract/Loss_evaluated_at', 666.667, 6000.0),
            ('/bilateral_syrinx/5_Tract/Beak_lowpass', 2666.67, 24000.0),
            ('/bilateral_syrinx/5_Tract/Beak_reflection', 0.9, 1.0),
            ('/bilateral_syrinx/5_Tract/End_correction_factor', 0.2, 1.8),
            ('/bilateral_syrinx/5_Tract/Junction_loss', 0.0, 0.2),
            ('/bilateral_syrinx/6_Output/Master_gain', -60.0, 0.0),
            ('/bilateral_syrinx/7_Valve_structure/Bernoulli_C_scale', 0.333333, 3.0),
            ('/bilateral_syrinx/7_Valve_structure/Contact_threshold', -0.2, 0.2),
            ('/bilateral_syrinx/7_Valve_structure/Tissue_mass_exponent', 1.0, 3.0),
            ('/bilateral_syrinx/7_Valve_structure/Rest_opening_share_to_mode_2', -1.0, 1.0),
            ('/bilateral_syrinx/7_Valve_structure/Control_smoothing', 1.0, 9.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_exc_bias', -8.4, 1.6),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_exc_from_HVC_burst', 0.0, 10.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_exc_self', 1.0, 11.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_exc_from_RA_inh', -8.0, 2.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_inh_bias', -12.0, -2.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_inh_from_HVC_stop_burst', -5.0, 5.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_inh_from_RA_exc', 1.0, 11.0),
            ('/bilateral_syrinx/A_Syllable/1_RA/RA_inh_self', -2.0, 8.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_exc_from_RA', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_exc_from_IA_pulse', -4.0, 6.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_exc_self', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_exc_from_ER_inh', -6.1, 3.9),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_inh_bias', -16.5, -6.5),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_inh_from_RA', -5.0, 5.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_inh_from_ER_exc', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_inh_self', -3.0, 7.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/IR_bias', -5.0, 5.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/IR_from_RA', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/IR_from_ER_exc', -15.0, -5.0),
            ('/bilateral_syrinx/A_Syllable/2_Respiratory/ER_exc_bias', -12.45, -2.45),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vs_bias', -8.0, 2.0),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vs_from_IA_pulse', -4.0, 6.0),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vs_from_RA', -4.5, 5.5),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vs_from_ER', -4.5, 5.5),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vs_from_IR', -5.0, 5.0),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vtb_bias', -8.0, 2.0),
            ('/bilateral_syrinx/A_Syllable/3_nXII/vtb_from_IR', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Pressure_gain', 0.666667, 6.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Pressure_offset', -2.0, 2.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Tension_L_gain', 9.66667, 87.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Tension_L_offset', -10.0, 10.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Tension_R_gain', 9.5, 85.5),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Tension_R_offset', -10.0, 10.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Gating_L_vtb_gain', 13.3333, 120.0),
            ('/bilateral_syrinx/A_Syllable/4_Scaling/Gating_R_vtb_gain', 0.0, 20.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/Syllable_rate', 0.95, 8.55),
            ('/bilateral_syrinx/A_Syllable/5_Timing/IA_pulse_width', 6.66667, 60.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/IA_to_nXII_delay', 3.33333, 30.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/IA_to_HVC_to_RA_delay', 10.0, 90.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/HVC_burst_width', 10.0, 90.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/HVC_stop_burst_delay', 100.0, 900.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/HVC_periodic_amount', 0.0, 10.0),
            ('/bilateral_syrinx/A_Syllable/5_Timing/HVC_periodic_rate', 8.33333, 75.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/dtb_L_bias', -8.0, 2.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/dtb_L_from_IA_pulse', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/dtb_L_from_RA', -5.0, 5.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/dtb_R_bias', -8.0, 2.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/dtb_R_from_IA_pulse', -5.0, 5.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/dtb_R_from_RA', 5.0, 15.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/Gating_L_dtb_gain', 13.3333, 120.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/Gating_L_offset', -10.0, 10.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/Gating_R_dtb_gain', 6.66667, 60.0),
            ('/bilateral_syrinx/A_Syllable/6_Bilateral/Gating_R_offset', -3.0, 17.0),
            ('/bilateral_syrinx/B_Mapping/Pressure_scale', 333.333, 3000.0),
            ('/bilateral_syrinx/B_Mapping/Tension_reference_frequency', 500.0, 4500.0),
            ('/bilateral_syrinx/B_Mapping/Tension_reference_value', 9.66667, 87.0),
            ('/bilateral_syrinx/B_Mapping/Gating_to_opening', 0.0333333, 0.3),
        ),
    }
)


def birdsong_synth_parameters(identity: ParamSpecName) -> list[ContinuousParameter]:
    """Build fresh exact-address controls for one birdsong model.

    :param identity: Registered birdsong parameter-spec identity.
    :returns: Continuous controls in compiler order.
    """
    return [
        ContinuousParameter(name=name, min=minimum, max=maximum)
        for name, minimum, maximum in BIRDSONG_PARAMETER_DOMAINS[identity]
    ]

_RIGHT_ROUTING_TOKEN = "__RIGHT_ROUTING__"
_SINGLE_RIGHT_ROUTING = """p46 = -3;
p47 = 0;
p48 = 10;"""
_BILATERAL_RIGHT_ROUTING = """p46 = hslider("h:A Syllable/h:6 Bilateral/[47]dtb R bias", -3, -8, 2, 0.01);
p47 = hslider("h:A Syllable/h:6 Bilateral/[48]dtb R from IA pulse", 0, -5, 5, 0.01);
p48 = hslider("h:A Syllable/h:6 Bilateral/[49]dtb R from RA", 10, 5, 15, 0.01);"""
_SYLLABLE_SOURCE_TEMPLATE = r'''// ---------------- syllable (52 static params, syllable.lib) and gesture mapping ----------------
syl = environment {
// syllable.lib — one canary syllable type as 52 static parameters, driving the
// song-system network of Alonso, Amador & Mindlin (2016). Output: five gestures
// P, T_L, T_R, G_L, G_R (paper units). Sliders are centred on the P0 tuple:
// weights/biases linear +-5, timings and gains log /3..x3, offsets linear.
// Use as:  sy = library("syllable.lib");  sy.gest : (P, TL, TR, GL, GR)


SR = fconstant(int fSamplingFreq, <math.h>);
T  = 1.0 / SR;
S(x) = 1.0 / (1.0 + exp(-x));
ms(x) = x * SR / 1000.0;


// ---- 1-8 RA ----
p1  = hslider("h:A Syllable/h:1 RA/[1]RA exc bias", -3.4, -8.4, 1.6, 0.01);
p2  = hslider("h:A Syllable/h:1 RA/[2]RA exc from HVC burst", 5, 0, 10, 0.01);
p3  = hslider("h:A Syllable/h:1 RA/[3]RA exc self", 6, 1, 11, 0.01);
p4  = hslider("h:A Syllable/h:1 RA/[4]RA exc from RA inh", -3, -8, 2, 0.01);
p5  = hslider("h:A Syllable/h:1 RA/[5]RA inh bias", -7, -12, -2, 0.01);
p6  = hslider("h:A Syllable/h:1 RA/[6]RA inh from HVC stop burst", 0, -5, 5, 0.01);
p7  = hslider("h:A Syllable/h:1 RA/[7]RA inh from RA exc", 6, 1, 11, 0.01);
p8  = hslider("h:A Syllable/h:1 RA/[8]RA inh self", 3, -2, 8, 0.01);
// ---- 9-20 respiratory ----
p9  = hslider("h:A Syllable/h:2 Respiratory/[9]ER exc bias", -7.45, -12.45, -2.45, 0.01);
p10 = hslider("h:A Syllable/h:2 Respiratory/[10]ER exc from RA", 10, 5, 15, 0.01);
p11 = hslider("h:A Syllable/h:2 Respiratory/[11]ER exc from IA pulse", 1, -4, 6, 0.01);
p12 = hslider("h:A Syllable/h:2 Respiratory/[12]ER exc self", 10, 5, 15, 0.01);
p13 = hslider("h:A Syllable/h:2 Respiratory/[13]ER exc from ER inh", -1.1, -6.1, 3.9, 0.01);
p14 = hslider("h:A Syllable/h:2 Respiratory/[14]ER inh bias", -11.5, -16.5, -6.5, 0.01);
p15 = hslider("h:A Syllable/h:2 Respiratory/[15]ER inh from RA", 0, -5, 5, 0.01);
p16 = hslider("h:A Syllable/h:2 Respiratory/[16]ER inh from ER exc", 10, 5, 15, 0.01);
p17 = hslider("h:A Syllable/h:2 Respiratory/[17]ER inh self", 2, -3, 7, 0.01);
p18 = hslider("h:A Syllable/h:2 Respiratory/[18]IR bias", 0, -5, 5, 0.01);
p19 = hslider("h:A Syllable/h:2 Respiratory/[19]IR from RA", 10, 5, 15, 0.01);
p20 = hslider("h:A Syllable/h:2 Respiratory/[20]IR from ER exc", -10, -15, -5, 0.01);
// ---- 21-27 tension and abductor pools ----
p21 = hslider("h:A Syllable/h:3 nXII/[21]vs bias", -3, -8, 2, 0.01);
p22 = hslider("h:A Syllable/h:3 nXII/[22]vs from IA pulse", 1, -4, 6, 0.01);
p23 = hslider("h:A Syllable/h:3 nXII/[23]vs from RA", 0.5, -4.5, 5.5, 0.01);
p24 = hslider("h:A Syllable/h:3 nXII/[24]vs from ER", 0.5, -4.5, 5.5, 0.01);
p25 = hslider("h:A Syllable/h:3 nXII/[25]vs from IR", 0, -5, 5, 0.01);
p26 = hslider("h:A Syllable/h:3 nXII/[26]vtb bias", -3, -8, 2, 0.01);
p27 = hslider("h:A Syllable/h:3 nXII/[27]vtb from IR", 10, 5, 15, 0.01);
// ---- 28-34 scalings ----
p28 = hslider("h:A Syllable/h:4 Scaling/[28]Pressure gain[scale:log]", 2, 0.666667, 6, 0.00666667);
p29 = hslider("h:A Syllable/h:4 Scaling/[29]Pressure offset", 0, -2, 2, 0.001);
p30 = hslider("h:A Syllable/h:4 Scaling/[30]Tension L gain[scale:log]", 29, 9.66667, 87, 0.0966667);
p31 = hslider("h:A Syllable/h:4 Scaling/[31]Tension L offset", 0, -10, 10, 0.001);
p32 = hslider("h:A Syllable/h:4 Scaling/[32]Tension R gain[scale:log]", 28.5, 9.5, 85.5, 0.095);
p33 = hslider("h:A Syllable/h:4 Scaling/[33]Tension R offset", 0, -10, 10, 0.001);
p34L = hslider("h:A Syllable/h:4 Scaling/[34]Gating L vtb gain[scale:log]", 40, 13.3333, 120, 0.133333);
p34R = hslider("h:A Syllable/h:4 Scaling/[35]Gating R vtb gain", 0, 0, 20, 0.01);      // canonical 0: floor, not centred
// ---- 35-42 timing ----
rate  = hslider("h:A Syllable/h:5 Timing/[36]Syllable rate [unit:Hz][scale:log]", 2.85, 0.95, 8.55, 0.0095);
iaw   = hslider("h:A Syllable/h:5 Timing/[37]IA pulse width [unit:ms][scale:log]", 20, 6.66667, 60, 0.0666667);
dn    = hslider("h:A Syllable/h:5 Timing/[38]IA to nXII delay [unit:ms][scale:log]", 10, 3.33333, 30, 0.0333333);
dh    = hslider("h:A Syllable/h:5 Timing/[39]IA to HVC to RA delay [unit:ms][scale:log]", 30, 10, 90, 0.1);
hw    = hslider("h:A Syllable/h:5 Timing/[40]HVC burst width [unit:ms][scale:log]", 30, 10, 90, 0.1);
stopd = hslider("h:A Syllable/h:5 Timing/[41]HVC stop burst delay [unit:ms][scale:log]", 300, 100, 900, 1);
pamt  = hslider("h:A Syllable/h:5 Timing/[42]HVC periodic amount", 0, 0, 10, 0.01);   // canonical 0: floor
prate = hslider("h:A Syllable/h:5 Timing/[43]HVC periodic rate [unit:Hz][scale:log]", 25, 8.33333, 75, 0.0833333);
// ---- 43-52 bilateral routing ----
p43 = hslider("h:A Syllable/h:6 Bilateral/[44]dtb L bias", -3, -8, 2, 0.01);
p44 = hslider("h:A Syllable/h:6 Bilateral/[45]dtb L from IA pulse", 10, 5, 15, 0.01);
p45 = hslider("h:A Syllable/h:6 Bilateral/[46]dtb L from RA", 0, -5, 5, 0.01);
__RIGHT_ROUTING__
p49 = hslider("h:A Syllable/h:6 Bilateral/[50]Gating L dtb gain[scale:log]", 40, 13.3333, 120, 0.133333);
p50 = hslider("h:A Syllable/h:6 Bilateral/[51]Gating L offset", 0, -10, 10, 0.001);
p51 = hslider("h:A Syllable/h:6 Bilateral/[52]Gating R dtb gain[scale:log]", 20, 6.66667, 60, 0.0666667);
p52 = hslider("h:A Syllable/h:6 Bilateral/[53]Gating R offset", 7, -3, 17, 0.001);

// ---- forcing ----
clock = ba.pulse(max(1, int(SR / rate)));
age   = (_ + 1 : *(1 - clock)) ~ _;
pulse(start, width) = 10.0 * ((age >= ms(start)) & (age < ms(start + width)));
inwin(start, width) = (age >= ms(start)) & (age < ms(start + width));
F   = pulse(0, iaw);
Fn  = pulse(dn, iaw);
Fd  = pulse(dh, hw) + pamt * 0.5 * (1 + os.osc(prate)) * inwin(dh, hw);
Fd2 = pulse(stopd, hw);

// ---- network: 9 rate ODEs, forward Euler ----
neural = (step ~ si.bus(9))
with {
    step(era, ira, eer, ier, eir, evs, edl, edr, evtb) =
        era  + T*20 *(-era  + S(p1  + p2*Fd  + p3*era + p4*ira)),
        ira  + T*20 *(-ira  + S(p5  + p6*Fd2 + p7*era + p8*ira)),
        eer  + T*250*(-eer  + S(p9  + p10*era + p11*F + p12*eer + p13*ier)),
        ier  + T*250*(-ier  + S(p14 + p15*era + p16*eer + p17*ier)),
        eir  + T*250*(-eir  + S(p18 + p19*era + p20*eer)),
        evs  + T*250*(-evs  + S(p21 + p22*Fn + p23*era + p24*eer + p25*eir)),
        edl  + T*250*(-edl  + S(p43 + p44*Fn + p45*era)),
        edr  + T*250*(-edr  + S(p46 + p47*Fn + p48*era)),
        evtb + T*250*(-evtb + S(p26 + p27*eir));
};
gest = neural : (!, !, _, !, _, _, _, _, _) : g
with {
    g(eer, eir, evs, edl, edr, evtb) = p28*eer + p29, p30*evs + p31, p32*evs + p33, p49*edl - p34L*evtb + p50, p51*edr - p34R*evtb + p52;
};

};
'''


_SINGLE_SYRINX_SOURCE = (
    r'''// single_syrinx.dsp — single-side syringeal valve + trachea, after
// N. H. Fletcher, "Bird song — a quantitative acoustic model",
// J. theor. Biol. 135 (1988) 455-481, as implemented by Smyth & Smith (2002).
//
// All equations from Fletcher's typeset text and Appendix A:
//   eq 1   dp0/dt = (rho c^2/V) ((pG - p0)/ZG - U)
//   eq 2,3 p0 - p1 = C U|U| + D dU/dt,  C = rho/(8 a^2 x^2),  D = rho/(2 sqrt(a x))
//   eq 8   m_n [x_n'' + 2 kappa x_n' + w_n^2 (x_n - x0)] = eps F
//   eq 9   m_1 = A3 rho_M pi a h d / 4
//   eq 10  kappa -> E kappa when x <= 0
//   eq 11  m -> m (1 + eta ((x - x0)/h)^2)
//   A6     F = A1 a h (p0 + p1) - rho U^2 h/(4 x^2 (2a - x)) [ sqrt(x/(2a-x)) atan(sqrt((2a-x)/x)) + x/(2a) ]
//   eq 7   T = (A2/5) rho_M a h d w1^2   (tension in N <-> f1)
//   eq 15  beta = 1 - 2 alpha L, alpha = 2e-5 sqrt(w)/a ; end correction 0.6 b
// Trachea: waveguide, tracheal-side load p1 = Z0 U + 2 p-,  Z0 = rho c/(pi a^2).
// Defaults: Fletcher Table 1 (raven). Designed to run at 8x (352.8 kHz).

declare name "single_syrinx";
import("stdfaust.lib");

// NOTE: ma.SR is clamped to 192 kHz in Faust's maths.lib; read the true rate.
SR = fconstant(int fSamplingFreq, <math.h>);
T  = 1.0 / SR;
PI = ma.PI;


'''
    + _SYLLABLE_SOURCE_TEMPLATE.replace(_RIGHT_ROUTING_TOKEN, _SINGLE_RIGHT_ROUTING)
    + r'''KP   = hslider("h:B Mapping/[0]Pressure scale [unit:Pa per unit P][scale:log]", 1000, 1000/3, 3000, 1);
FREF = hslider("h:B Mapping/[1]Tension reference frequency [unit:Hz][scale:log]", 1500, 500, 4500, 1);
TREF = hslider("h:B Mapping/[2]Tension reference value[scale:log]", 29, 29/3, 87, 0.01);
KG   = hslider("h:B Mapping/[3]Gating to opening [unit:mm per unit G][scale:log]", 0.1, 0.1/3, 0.3, 0.0001);
gP  = syl.gest : (_,!,!,!,!);
gTL = syl.gest : (!,_,!,!,!);
gTR = syl.gest : (!,!,_,!,!);
gGL = syl.gest : (!,!,!,_,!);
gGR = syl.gest : (!,!,!,!,_);
f1of(Tn) = FREF * sqrt(max(0, Tn) / TREF);

// ---------------- environment ----------------
rho  = hslider("h:0 Air/[0]Air density [unit:kg/m3]", 1.2, 1.0, 1.4, 0.001);
c    = hslider("h:0 Air/[1]Speed of sound [unit:m/s]", 343, 300, 380, 0.1);

// ---------------- anatomy (Fletcher Table 1) ----------------
a    = hslider("h:1 Anatomy/[0]Bronchus radius a [unit:mm]", 3.5, 0.5, 10, 0.01) * 1e-3;
hh   = hslider("h:1 Anatomy/[1]Membrane half-width h [unit:mm]", 3.5, 0.5, 10, 0.01) * 1e-3;
dd   = hslider("h:1 Anatomy/[2]Membrane thickness d [unit:um]", 100, 10, 500, 1) * 1e-6;
rhoM = hslider("h:1 Anatomy/[3]Membrane density [unit:kg/m3]", 1000, 500, 1500, 1);
Lt   = hslider("h:1 Anatomy/[4]Trachea length L [unit:mm]", 70, 5, 120, 0.1) * 1e-3;
bb   = hslider("h:1 Anatomy/[5]Mouth horn radius b [unit:mm]", 10, 1, 20, 0.1) * 1e-3;
Vb   = hslider("h:1 Anatomy/[6]Bronchial volume V [unit:ml]", 1, 0.05, 5, 0.01) * 1e-6;
ZGk  = hslider("h:1 Anatomy/[7]Air-sac impedance ZG [unit:MPa.s/m3]", 2, 0.1, 20, 0.01) * 1e6;

// ---------------- membrane / operating ----------------
pG   = KP * gP : si.smoo;
f1   = max(20, f1of(gTL)) : si.smoo;
r2   = hslider("h:2 Valve/[2]Mode 2 ratio f2/f1", 1.667, 1, 4, 0.001);
kap  = hslider("h:2 Valve/[3]Damping kappa [unit:1/s]", 300, 10, 5000, 1);
E    = hslider("h:2 Valve/[4]Contact damping E", 30, 1, 100, 1);
eta  = hslider("h:2 Valve/[5]Tissue mass eta", 10, 0, 50, 0.1);
x0p  = hslider("h:2 Valve/[6]Phonation rest opening x0 [unit:mm]", 0, -2, 5, 0.01) * 1e-3;
x0   = x0p - KG * 1e-3 * abs(gGL) : si.smoo;
A1   = hslider("h:2 Valve/[7]Force constant A1", 1, 0.3, 3, 0.01);
A3   = hslider("h:2 Valve/[8]Mass constant A3", 1, 0.3, 3, 0.01);
eps1 = hslider("h:2 Valve/[9]Coupling eps mode 1", 1, 0, 1, 0.01);
eps2 = hslider("h:2 Valve/[10]Coupling eps mode 2", 1, 0, 1, 0.01);
sD   = hslider("h:2 Valve/[11]Inertance scale (0 = quasi-static)", 1, 0, 3, 0.01);
sS   = hslider("h:2 Valve/[12]Suction scale", 1, 0, 3, 0.01);

// ---------------- tract ----------------
alph = hslider("h:3 Tract/[0]Wall loss coefficient [unit:1e-5]", 2, 0, 10, 0.01) * 1e-5;
fal  = hslider("h:3 Tract/[1]Loss evaluated at [unit:Hz]", 1200, 100, 8000, 1);
fcB  = hslider("h:3 Tract/[2]Beak lowpass [unit:Hz]", 6000, 500, 20000, 1);
gain = hslider("h:4 Output/[0]Master gain [unit:dB]", -12, -60, 0, 0.1) : ba.db2linear;

// ---------------- derived ----------------
Z0   = rho * c / (PI * a * a);
w1   = 2 * PI * f1;
w2   = r2 * w1;
m1   = A3 * rhoM * PI * a * hh * dd / 4;
kV   = rho * c * c / Vb;
Leff = Lt + 0.6 * bb;
Ntr  = Leff / c * SR;
beta = max(0, 1 - 2 * (alph * sqrt(2 * PI * fal) / a) * Lt);
xmin = 1e-6;

// one membrane mode, implicit trapezoid rule for v' = F/m - 2 ge v - w^2 (x - xr)
mode(F, ge, m, w, xr, xm, vm, am) = xn, vn, an
with {
    den = 1 + ge*T + w*w*T*T/4;
    vn  = (vm + T/2*am + T/2*(F/m - w*w*(xm - xr)) - w*w*T*T/4*vm) / den;
    xn  = xm + T/2*(vm + vn);
    an  = F/m - 2*ge*vn - w*w*(xn - xr);
};

// state: p0, U, (x,v,a) x 2 modes, pm ; outputs: state..., out, p1, U, x, p0
step(p0, U, x1, v1, a1, x2, v2, a2, pm) =
    p0n, Un,
    mode(eps1*F, ge, mass, w1, x0, x1, v1, a1),
    mode(eps2*F, ge, mass, w2, 0,  x2, v2, a2),
    pmn,
    out, p1, U, x, p0
with {
    x   = x1 + x2;
    op  = x > 0;
    xc  = max(xmin, min(x, 2*a - xmin));

    // flow, eq 2-3, implicit-linearised in U (unconditionally stable)
    Cq  = rho / (8 * a * a * xc * xc);
    Dq  = max(1e-3, sD * rho / (2 * sqrt(a * xc)));
    dp  = p0 - 2*pm;                                    // p1 = Z0 U + 2 pm
    Un  = op * (U + T*(dp + Cq*U*abs(U)) / Dq) / (1 + T*(Z0 + 2*Cq*abs(U)) / Dq);
    p1  = Z0 * Un + 2*pm;

    // membrane force, Appendix A6 (full form)
    r   = xc / (2*a - xc);
    suc = rho*Un*Un*hh / (4*xc*xc*(2*a - xc)) * (sqrt(r) * atan(sqrt(1/r)) + xc/(2*a));
    F   = A1 * a * hh * (p0 + p1) - op * sS * suc;
    ge  = kap * (1 + (E - 1) * (x <= 0));
    mass = m1 * (1 + eta * ((x - x0)/hh) * ((x - x0)/hh));

    // bronchial pressure, eq 1, backward Euler
    p0n = (p0 + T*kV*(pG/ZGk - Un)) / (1 + T*kV/ZGk);

    // trachea
    pplus = Z0 * Un + pm;
    pmn = pplus : de.fdelay(4096, max(1, 2*Ntr - 1)) : fi.lowpass(1, fcB) : *(-beta);
    out = pplus : de.fdelay(4096, Ntr) <: _ - (fi.lowpass(1, fcB) : *(beta));
};

safe(x) = select2(x == x, 0, x) : max(-1e6) : min(1e6) : ma.tanh : *(0.891);

// output 0: radiated (scaled, safe); taps 1..4: p1 [Pa], U [m3/s], x [m], p0 [Pa]
process = (step ~ si.bus(9)) : (si.block(9), (*(2e-4*gain) : safe), _, _, _, _) : (_, !, !, !, !);
'''
)

_BILATERAL_SYRINX_SOURCE = (
    r'''// bilateral_syrinx.dsp — bilateral avian vocal tract: two Fletcher (1988) syringeal
// valves, two bronchial waveguides, the 3-port parallel junction of
// Smyth & Smith (ASA 2002, eqs 18-20), a tracheal waveguide and a beak.
//
//   valve L -> bronchus L --\
//                            3-port --> trachea --> beak (radiates)
//   valve R -> bronchus R --/
//
// Junction (lossless, admittance-weighted):
//   P_J = 2 * sum(G_i P_i+) / sum(G_i),   P_i- = P_J - P_i+,   G_i = pi r_i^2 / (rho c)
// Each valve sees its bronchus:  p1 = Z_b U + 2 p-   (Z_b = rho c / (pi a^2)).
// Valve equations exactly as single_syrinx.dsp (Fletcher eq 1-3, 8-11, App. A6).
// Designed for 8x oversampling (352.8 kHz); reads the true sample rate.
// Input 0: test signal injected into the trachea at the junction (for
// passivity checks); leave unconnected for normal use.
// Outputs: 0 radiated, 1 p_J, 2 p1 L, 3 p1 R, 4 U L, 5 U R, 6 x L, 7 x R.

declare name "bilateral_syrinx";
import("stdfaust.lib");

SR = fconstant(int fSamplingFreq, <math.h>);
T  = 1.0 / SR;
PI = ma.PI;

// ---------------- sliders: canonical value at the slider midpoint ----------------
// [lit] literature value, [derived] scaled from literature, [choice] ours.
// Log sliders span canonical/3 .. canonical*3 (Fletcher's "factor of order unity");
// linear sliders are symmetric about the canonical value.

'''
    + _SYLLABLE_SOURCE_TEMPLATE.replace(_RIGHT_ROUTING_TOKEN, _BILATERAL_RIGHT_ROUTING)
    + r'''KP   = hslider("h:B Mapping/[0]Pressure scale [unit:Pa per unit P][scale:log]", 1000, 1000/3, 3000, 1);
FREF = hslider("h:B Mapping/[1]Tension reference frequency [unit:Hz][scale:log]", 1500, 500, 4500, 1);
TREF = hslider("h:B Mapping/[2]Tension reference value[scale:log]", 29, 29/3, 87, 0.01);
KG   = hslider("h:B Mapping/[3]Gating to opening [unit:mm per unit G][scale:log]", 0.1, 0.1/3, 0.3, 0.0001);
gP  = syl.gest : (_,!,!,!,!);
gTL = syl.gest : (!,_,!,!,!);
gTR = syl.gest : (!,!,_,!,!);
gGL = syl.gest : (!,!,!,_,!);
gGR = syl.gest : (!,!,!,!,_);
f1of(Tn) = FREF * sqrt(max(0, Tn) / TREF);

// ---------------- environment ----------------
rho  = hslider("h:0 Air/[0]Air density [unit:kg/m3]", 1.2, 1.0, 1.4, 0.001);              // [lit]
c    = hslider("h:0 Air/[1]Speed of sound [unit:m/s]", 343, 300, 386, 0.1);               // [lit]

// ---------------- anatomy, per side (canonical: SMAC-03 medium bird) ----------------
aL   = hslider("h:1 Anatomy L/[0]Bronchus radius [unit:mm][scale:log]", 2.5, 2.5/3, 7.5, 0.01) * 1e-3;       // [lit]
LbL  = hslider("h:1 Anatomy L/[1]Bronchus length [unit:mm][scale:log]", 14, 14/3, 42, 0.01) * 1e-3;          // [lit]
hL   = hslider("h:1 Anatomy L/[2]Membrane half-width [unit:mm][scale:log]", 1.75, 1.75/3, 5.25, 0.01) * 1e-3; // [derived] half of Smyth's 3.5 mm
VL   = hslider("h:1 Anatomy L/[3]Lower bronchus volume [unit:ml][scale:log]", 0.27, 0.09, 0.81, 0.001) * 1e-6; // [derived] pi a^2 Lb
ZGL  = hslider("h:1 Anatomy L/[4]Air-sac impedance [unit:MPa.s/m3][scale:log]", 2, 2/3, 6, 0.01) * 1e6;     // [derived] ~0.1 Zb, Fletcher Fig 4
aR   = hslider("h:2 Anatomy R/[0]Bronchus radius [unit:mm][scale:log]", 2.5, 2.5/3, 7.5, 0.01) * 1e-3;
LbR  = hslider("h:2 Anatomy R/[1]Bronchus length [unit:mm][scale:log]", 14, 14/3, 42, 0.01) * 1e-3;
hR   = hslider("h:2 Anatomy R/[2]Membrane half-width [unit:mm][scale:log]", 1.75, 1.75/3, 5.25, 0.01) * 1e-3;
VR   = hslider("h:2 Anatomy R/[3]Lower bronchus volume [unit:ml][scale:log]", 0.27, 0.09, 0.81, 0.001) * 1e-6;
ZGR  = hslider("h:2 Anatomy R/[4]Air-sac impedance [unit:MPa.s/m3][scale:log]", 2, 2/3, 6, 0.01) * 1e6;
ddL  = hslider("h:1 Anatomy L/[5]Membrane thickness [unit:um][scale:log]", 100, 100/3, 300, 0.1) * 1e-6;   // [lit]
rhoML= hslider("h:1 Anatomy L/[6]Membrane density [unit:kg/m3][scale:log]", 1000, 500, 2000, 1);            // [lit]
ddR  = hslider("h:2 Anatomy R/[5]Membrane thickness [unit:um][scale:log]", 100, 100/3, 300, 0.1) * 1e-6;
rhoMR= hslider("h:2 Anatomy R/[6]Membrane density [unit:kg/m3][scale:log]", 1000, 500, 2000, 1);
Lt   = hslider("h:3 Anatomy shared/[2]Trachea length [unit:mm][scale:log]", 23, 23/3, 69, 0.01) * 1e-3;      // [lit]
at   = hslider("h:3 Anatomy shared/[3]Trachea radius [unit:mm][scale:log]", 3.5, 3.5/3, 10.5, 0.01) * 1e-3;  // [lit]
bb   = hslider("h:3 Anatomy shared/[4]Mouth horn radius [unit:mm][scale:log]", 10, 10/3, 30, 0.01) * 1e-3;   // [derived] Fletcher b = 2.9 a

// ---------------- valves ----------------
pG   = KP * gP : sm;
f1L  = max(20, f1of(gTL)) : sm;
f1R  = max(20, f1of(gTR)) : sm;
x0pL = hslider("h:4 Valves/[3]Phonation rest opening L [unit:mm]", 0, -1.5, 1.5, 0.01) * 1e-3;
x0L  = x0pL - KG * 1e-3 * abs(gGL) : sm;
x0pR = hslider("h:4 Valves/[4]Phonation rest opening R [unit:mm]", 0, -1.5, 1.5, 0.01) * 1e-3;
x0R  = x0pR - KG * 1e-3 * abs(gGR) : sm;
r2   = hslider("h:4 Valves/[5]Mode 2 ratio f2/f1", 1.6, 1.2, 2.0, 0.001);                                 // [lit] Fletcher ~1.6
kap  = hslider("h:4 Valves/[6]Damping kappa [unit:1/s][scale:log]", 1571, 1571/3, 4713, 1);              // [derived] Q=2 at 1 kHz
E    = hslider("h:4 Valves/[7]Contact damping E[scale:log]", 31.6, 10, 100, 0.1);                         // [lit] 10..100
eta  = hslider("h:4 Valves/[8]Tissue mass eta[scale:log]", 10, 10/3, 30, 0.01);                          // [lit] ~10
A1   = hslider("h:4 Valves/[9]Force constant A1[scale:log]", 1, 1/3, 3, 0.001);                          // [lit] order unity
A3   = hslider("h:4 Valves/[10]Mass constant A3[scale:log]", 1, 1/3, 3, 0.001);                          // [lit] order unity
eps1 = hslider("h:4 Valves/[11]Coupling eps mode 1", 1, 0.5, 1.5, 0.001);                                 // [lit] 1
eps2 = hslider("h:4 Valves/[12]Coupling eps mode 2", 1, 0.5, 1.5, 0.001);
sD   = hslider("h:4 Valves/[13]Inertance scale[scale:log]", 1, 1/3, 3, 0.001);                            // [lit] D to order unity
sS   = hslider("h:4 Valves/[14]Suction scale[scale:log]", 1, 1/3, 3, 0.001);                              // [lit] A1 order unity

// ---------------- tract ----------------
alph = hslider("h:5 Tract/[0]Wall loss coefficient [unit:1e-5][scale:log]", 2, 2/3, 6, 0.001) * 1e-5;    // [lit] Fletcher 2e-5
fal  = hslider("h:5 Tract/[1]Loss evaluated at [unit:Hz][scale:log]", 2000, 2000/3, 6000, 1);            // [choice]
fcB  = hslider("h:5 Tract/[2]Beak lowpass [unit:Hz][scale:log]", 8000, 8000/3, 24000, 1);               // [choice]
rB   = hslider("h:5 Tract/[3]Beak reflection", 0.95, 0.9, 1.0, 0.001);                                    // [choice]
endc = hslider("h:5 Tract/[4]End correction factor[scale:log]", 0.6, 0.2, 1.8, 0.001);                 // [lit] Fletcher 0.6 b
jl   = hslider("h:5 Tract/[5]Junction loss", 0, 0, 0.2, 0.001);                                           // [choice] 0 = lossless (ASA-02); not centred, 0 is the physical floor

// ---------------- valve structure constants (Fletcher's fixed forms, exposed) ----------------
sC   = hslider("h:7 Valve structure/[0]Bernoulli C scale[scale:log]", 1, 1/3, 3, 0.001);                // [lit] C = rho/(8 a^2 x^2), order unity
xce  = hslider("h:7 Valve structure/[1]Contact threshold [unit:mm]", 0, -0.2, 0.2, 0.001) * 1e-3;       // [choice] x <= xce triggers E kappa
etax = hslider("h:7 Valve structure/[2]Tissue mass exponent", 2, 1, 3, 0.01);                            // [lit] Fletcher eq 11 uses 2
x0m2 = hslider("h:7 Valve structure/[3]Rest opening share to mode 2", 0, -1, 1, 0.01);                   // [choice] 0 = x0 on mode 1 only
smt  = hslider("h:7 Valve structure/[4]Control smoothing [unit:ms][scale:log]", 3, 1, 9, 0.01);          // [choice] slew on pG, f1, x0

gain = hslider("h:6 Output/[0]Master gain [unit:dB]", -12, -60, 0, 0.1) : ba.db2linear;
tin  = hslider("h:6 Output/[1]Test input gain", 0, 0, 1, 0.01);

// ---------------- derived ----------------
sm = si.smooth(ba.tau2pole(smt * 1e-3));
Zb(a) = rho * c / (PI * a * a);
G(r)  = PI * r * r / (rho * c);
w0 = 2 * PI * fal;
loss(L, r) = exp(0 - (alph * sqrt(w0) / r) * L);           // one-way wall loss per tube
NbL = max(1, LbL / c * SR);
NbR = max(1, LbR / c * SR);
Ntr = max(1, (Lt + endc * bb) / c * SR);
xmin = 1e-6;
DMAX = 8192;

mode(F, ge, m, w, xr, xm, vm, am) = xn, vn, an
with {
    den = 1 + ge*T + w*w*T*T/4;
    vn  = (vm + T/2*am + T/2*(F/m - w*w*(xm - xr)) - w*w*T*T/4*vm) / den;
    xn  = xm + T/2*(vm + vn);
    an  = F/m - 2*ge*vn - w*w*(xn - xr);
};

// one Fletcher valve. inputs: state (p0,U,x1,v1,a1,x2,v2,a2), returning wave pm
// outputs: new state (8), pplus (wave launched into bronchus), p1, U, x
valve(a, hh, V, ZG, dd, rhoM, f1, x0, p0, U, x1, v1, a1, x2, v2, a2, pm) =
    p0n, Un, mode(eps1*F, ge, mass, w1, x0, x1, v1, a1), mode(eps2*F, ge, mass, w2, x0m2*x0, x2, v2, a2),
    pplus, p1, Un, x
with {
    Z0  = Zb(a);
    w1  = 2 * PI * f1;  w2 = r2 * w1;
    m1  = A3 * rhoM * PI * a * hh * dd / 4;
    kV  = rho * c * c / V;
    x   = x1 + x2;
    op  = x > 0;
    xc  = max(xmin, min(x, 2*a - xmin));
    Cq  = sC * rho / (8 * a * a * xc * xc);
    Dq  = max(1e-3, sD * rho / (2 * sqrt(a * xc)));
    dp  = p0 - 2*pm;
    Un  = op * (U + T*(dp + Cq*U*abs(U)) / Dq) / (1 + T*(Z0 + 2*Cq*abs(U)) / Dq);
    p1  = Z0 * Un + 2*pm;
    r   = xc / (2*a - xc);
    suc = rho*Un*Un*hh / (4*xc*xc*(2*a - xc)) * (sqrt(r) * atan(sqrt(1/r)) + xc/(2*a));
    F   = A1 * a * hh * (p0 + p1) - op * sS * suc;
    ge  = kap * (1 + (E - 1) * (x <= xce));
    mass = m1 * (1 + eta * pow(abs((x - x0)/hh), etax));
    p0n = (p0 + T*kV*(pG/ZG - Un)) / (1 + T*kV/ZG);
    pplus = Z0 * Un + pm;
};

// full step. state bus: L(8), R(8), pmL, pmR, pTp  = 19 ; plus test input
step(sL1,sL2,sL3,sL4,sL5,sL6,sL7,sL8, sR1,sR2,sR3,sR4,sR5,sR6,sR7,sR8, pmL, pmR, pTp, xin) =
    nL, nR, pmLn, pmRn, pTpn, out, pJ, p1L, p1R, UL, UR, xL, xR
with {
    vL = valve(aL, hL, VL, ZGL, ddL, rhoML, f1L, x0L, sL1,sL2,sL3,sL4,sL5,sL6,sL7,sL8, pmL);
    vR = valve(aR, hR, VR, ZGR, ddR, rhoMR, f1R, x0R, sR1,sR2,sR3,sR4,sR5,sR6,sR7,sR8, pmR);
    nL = vL : si.bus(8), si.block(4);
    nR = vR : si.bus(8), si.block(4);
    ppL = vL : si.block(8), _, !, !, !;   p1L = vL : si.block(9), _, !, !;   UL = vL : si.block(10), _, !;   xL = vL : si.block(11), _;
    ppR = vR : si.block(8), _, !, !, !;   p1R = vR : si.block(9), _, !, !;   UR = vR : si.block(10), _, !;   xR = vR : si.block(11), _;

    gL = loss(LbL, aL);  gR = loss(LbR, aR);  gT = loss(Lt, at);
    GL = G(aL);  GR = G(aR);  GT = G(at);

    // waves arriving at the junction (bronchi: one-way delay + loss)
    pLj = ppL : de.fdelay(DMAX, NbL) : *(gL);
    pRj = ppR : de.fdelay(DMAX, NbR) : *(gR);
    pJ  = (1 - jl) * 2 * (GL*pLj + GR*pRj + GT*pTp) / (GL + GR + GT) + tin * xin;

    // waves leaving the junction
    pmLn = (pJ - pLj) : de.fdelay(DMAX, NbL - 1) : *(gL);              // back down to valve L
    pmRn = (pJ - pRj) : de.fdelay(DMAX, NbR - 1) : *(gR);
    pTup = (pJ - pTp) * gT;                                             // up the trachea
    pTpn = pTup : de.fdelay(DMAX, 2*Ntr - 1) : fi.lowpass(1, fcB) : *(0 - rB * gT);  // beak reflection, back to junction
    out  = pTup : de.fdelay(DMAX, Ntr) <: _ - (fi.lowpass(1, fcB) : *(rB));            // radiated
};

safe(x) = select2(x == x, 0, x) : max(-1e6) : min(1e6) : ma.tanh : *(0.891);

process = 0 : (step ~ si.bus(19)) : (si.block(19), (*(2e-3 * gain) : safe), _, _, _, _, _, _, _) : (_, !, !, !, !, !, !, !);
'''
)
