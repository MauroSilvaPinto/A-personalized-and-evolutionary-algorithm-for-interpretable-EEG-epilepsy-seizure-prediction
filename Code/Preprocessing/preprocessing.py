"""

This code was made on Matlab, with the intention of preprocessing the data,
and to extract first_level features.

We extracted features by iterating windows of 5 seconds that
do not overlap. The signal is the EEG, 10-20 system, fs sampling of 256.

variables beginning_index:ending_index are indexes that concern intervals
separated by 256*5 samples (5 seconds)

                %% Frequency Analysis
                % raw EEG : 0   - 45 Hz
                % delta   : 0.5 - 3.5 Hz
                % theta   : 4   - 7.5 Hz
                % alpha   : 8   - 12 Hz
                % beta    : 13  - 35 Hz
                
                fft_signal=fft(signal(beginning_index:ending_index));
                psd_signal=fft_signal.^2;
                L=length(fft_signal);
                
                psd_signal=psd_signal(1:round(L/2));
                frequency_vector=linspace(0,fs/2,length(psd_signal));
                
                % frequency band waves
                delta_signal=psd_signal(intersect(find(frequency_vector>=0.5), find(frequency_vector<=3.5)));
                theta_signal=psd_signal(intersect(find(frequency_vector>=4), find(frequency_vector<=7.5)));
                alpha_signal=psd_signal(intersect(find(frequency_vector>=8), find(frequency_vector<=12)));
                beta_signal=psd_signal(intersect(find(frequency_vector>=13), find(frequency_vector<=35)));
                gamma_signal=psd_signal(intersect(find(frequency_vector>=36), find(frequency_vector<=50)));
                gamma_signal_2=psd_signal(intersect(find(frequency_vector>=50), find(frequency_vector<=70)));
                gamma_signal_3=psd_signal(intersect(find(frequency_vector>=70), find(frequency_vector<=90)));
                
                
                % calculating the power (area below curves)
                total_power=trapz(abs(psd_signal));
                if total_power~=0
                    delta_power=trapz(abs((delta_signal)));
                    theta_power=trapz(abs((theta_signal)));
                    alpha_power=trapz(abs(alpha_signal));
                    beta_power=trapz(abs((beta_signal)));
                    gamma_power=trapz(abs(gamma_signal));
                    gamma_power_2=trapz(abs(gamma_signal_2));
                    gamma_power_3=trapz(abs(gamma_signal_3));
                else
                    delta_power=0;
                    theta_power=0;
                    alpha_power=0;
                    beta_power=0;
                    gamma_power=0;
                    gamma_power_2=0;
                    gamma_power_3=0;
                end
                
                
                % calculating medium intensity and variance, mean frequency and
                % bandpower
                medium_intensity=mean(abs(signal(beginning_index:ending_index))) ...
                    /max(abs(signal(beginning_index:ending_index)));
                
                medium_intensity_un=mean(abs(signal(beginning_index:ending_index)));
                
                variance=var(abs(signal(beginning_index:ending_index)));
                
                mean_freq=meanfreq(abs(psd_signal),fs);
                band_power=bandpower(abs(psd_signal),frequency_vector,'psd');
                
                
                % calculate normalized power frequencies and update vectors
                
                if total_power~=0
                    delta_signal_vector(kkk)=delta_power/total_power;
                    theta_signal_vector(kkk)=theta_power/total_power;
                    alpha_signal_vector(kkk)=alpha_power/total_power;
                    beta_signal_vector(kkk)=beta_power/total_power;
                    gamma_signal_vector(kkk)=gamma_power/total_power;
                    gamma_signal_vector_2(kkk)=gamma_power_2/total_power;
                    gamma_signal_vector_3(kkk)=gamma_power_3/total_power;
                else
                    delta_signal_vector(kkk)=0;
                    theta_signal_vector(kkk)=0;
                    alpha_signal_vector(kkk)=0;
                    beta_signal_vector(kkk)=0;
                    gamma_signal_vector(kkk)=0;
                    gamma_signal_vector_2(kkk)=0;
                    gamma_signal_vector_3(kkk)=0;
                end
                
                
                medium_signal_vector(kkk)=medium_intensity;
                variance_signal_vector(kkk)=variance;
                medium_signal_vector_un(kkk)=medium_intensity_un;
                
                mean_freq_vector(kkk)=mean_freq;
                band_power_vector(kkk)=band_power;
                
                
                
            end
            
            
            names=["delta_signal_vector",...
                "theta_signal_vector",...
                "alpha_signal_vector",...
                "beta_signal_vector",...
                "gamma_signal_vector",...
                "gamma_signal_vector_2",...
                "gamma_signal_vector_3",...
                "mean_freq_vector",...
                "band_power_vector",...
                "medium_intensity",...
                "medium_intensity_unormalized",...
                "variance"];
            
            features=[delta_signal_vector;...
                theta_signal_vector;...
                alpha_signal_vector;...
                beta_signal_vector;...
                gamma_signal_vector;...
                gamma_signal_vector_2;...
                gamma_signal_vector_3;...
                mean_freq_vector;...
                band_power_vector;...
                medium_signal_vector;...
                medium_signal_vector_un;...
                variance_signal_vector];
            

#########################################################


This code will extract the following features, for the following electrodes,
which will be stored in Processing_labels, by this order:
    
C3_delta
C3_theta
C3_alpha
C3_beta
C3_gamma_1
C3_gamma_2
C3_gamma_3
C3_mean_freq
C3_band_power
C3_medium_intensity
C3_medium_intensity_unormalized
C3_variance
C4_delta
C4_theta
C4_alpha
C4_beta
C4_gamma_1
C4_gamma_2
C4_gamma_3
C4_mean_freq
C4_band_power
C4_medium_intensity
C4_medium_intensity_unormalized
C4_variance
CZ_delta
CZ_theta
CZ_alpha
CZ_beta
CZ_gamma_1
CZ_gamma_2
CZ_gamma_3
CZ_mean_freq
CZ_band_power
CZ_medium_intensity
CZ_medium_intensity_unormalized
CZ_variance
F3_delta
F3_theta
F3_alpha
F3_beta
F3_gamma_1
F3_gamma_2
F3_gamma_3
F3_mean_freq
F3_band_power
F3_medium_intensity
F3_medium_intensity_unormalized
F3_variance
F4_delta
F4_theta
F4_alpha
F4_beta
F4_gamma_1
F4_gamma_2
F4_gamma_3
F4_mean_freq
F4_band_power
F4_medium_intensity
F4_medium_intensity_unormalized
F4_variance
F7_delta
F7_theta
F7_alpha
F7_beta
F7_gamma_1
F7_gamma_2
F7_gamma_3
F7_mean_freq
F7_band_power
F7_medium_intensity
F7_medium_intensity_unormalized
F7_variance
F8_delta
F8_theta
F8_alpha
F8_beta
F8_gamma_1
F8_gamma_2
F8_gamma_3
F8_mean_freq
F8_band_power
F8_medium_intensity
F8_medium_intensity_unormalized
F8_variance
FP1_delta
FP1_theta
FP1_alpha
FP1_beta
FP1_gamma_1
FP1_gamma_2
FP1_gamma_3
FP1_mean_freq
FP1_band_power
FP1_medium_intensity
FP1_medium_intensity_unormalized
FP1_variance
FP2_delta
FP2_theta
FP2_alpha
FP2_beta
FP2_gamma_1
FP2_gamma_2
FP2_gamma_3
FP2_mean_freq
FP2_band_power
FP2_medium_intensity
FP2_medium_intensity_unormalized
FP2_variance
FZ_delta
FZ_theta
FZ_alpha
FZ_beta
FZ_gamma_1
FZ_gamma_2
FZ_gamma_3
FZ_mean_freq
FZ_band_power
FZ_medium_intensity
FZ_medium_intensity_unormalized
FZ_variance
O1_delta
O1_theta
O1_alpha
O1_beta
O1_gamma_1
O1_gamma_2
O1_gamma_3
O1_mean_freq
O1_band_power
O1_medium_intensity
O1_medium_intensity_unormalized
O1_variance
O2_delta
O2_theta
O2_alpha
O2_beta
O2_gamma_1
O2_gamma_2
O2_gamma_3
O2_mean_freq
O2_band_power
O2_medium_intensity
O2_medium_intensity_unormalized
O2_variance
P3_delta
P3_theta
P3_alpha
P3_beta
P3_gamma_1
P3_gamma_2
P3_gamma_3
P3_mean_freq
P3_band_power
P3_medium_intensity
P3_medium_intensity_unormalized
P3_variance
P4_delta
P4_theta
P4_alpha
P4_beta
P4_gamma_1
P4_gamma_2
P4_gamma_3
P4_mean_freq
P4_band_power
P4_medium_intensity
P4_medium_intensity_unormalized
P4_variance
PZ_delta
PZ_theta
PZ_alpha
PZ_beta
PZ_gamma_1
PZ_gamma_2
PZ_gamma_3
PZ_mean_freq
PZ_band_power
PZ_medium_intensity
PZ_medium_intensity_unormalized
PZ_variance
RS_delta
RS_theta
RS_alpha
RS_beta
RS_gamma_1
RS_gamma_2
RS_gamma_3
RS_mean_freq
RS_band_power
RS_medium_intensity
RS_medium_intensity_unormalized
RS_variance
SP1_delta
SP1_theta
SP1_alpha
SP1_beta
SP1_gamma_1
SP1_gamma_2
SP1_gamma_3
SP1_mean_freq
SP1_band_power
SP1_medium_intensity
SP1_medium_intensity_unormalized
SP1_variance
SP2_delta
SP2_theta
SP2_alpha
SP2_beta
SP2_gamma_1
SP2_gamma_2
SP2_gamma_3
SP2_mean_freq
SP2_band_power
SP2_medium_intensity
SP2_medium_intensity_unormalized
SP2_variance
T1_delta
T1_theta
T1_alpha
T1_beta
T1_gamma_1
T1_gamma_2
T1_gamma_3
T1_mean_freq
T1_band_power
T1_medium_intensity
T1_medium_intensity_unormalized
T1_variance
T2_delta
T2_theta
T2_alpha
T2_beta
T2_gamma_1
T2_gamma_2
T2_gamma_3
T2_mean_freq
T2_band_power
T2_medium_intensity
T2_medium_intensity_unormalized
T2_variance
T3_delta
T3_theta
T3_alpha
T3_beta
T3_gamma_1
T3_gamma_2
T3_gamma_3
T3_mean_freq
T3_band_power
T3_medium_intensity
T3_medium_intensity_unormalized
T3_variance
T4_delta
T4_theta
T4_alpha
T4_beta
T4_gamma_1
T4_gamma_2
T4_gamma_3
T4_mean_freq
T4_band_power
T4_medium_intensity
T4_medium_intensity_unormalized
T4_variance
T5_delta
T5_theta
T5_alpha
T5_beta
T5_gamma_1
T5_gamma_2
T5_gamma_3
T5_mean_freq
T5_band_power
T5_medium_intensity
T5_medium_intensity_unormalized
T5_variance
T6_delta
T6_theta
T6_alpha
T6_beta
T6_gamma_1
T6_gamma_2
T6_gamma_3
T6_mean_freq
T6_band_power
T6_medium_intensity
T6_medium_intensity_unormalized
T6_variance

"""