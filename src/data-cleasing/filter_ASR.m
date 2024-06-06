% EEGLAB history file generated on the 06-Jun-2024
% ------------------------------------------------
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\USER\\Desktop\\CNELab\\2024_BCI_Intro\\More-features\\data.mat','srate',128,'pnts',0,'xmin',0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','data','gui','off'); 
EEG = pop_eegfiltnew(EEG, 'locutoff',0,'hicutoff',20,'plotfreqz',1);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','filtered data','gui','off'); 
EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'setname','ASR data','gui','off'); 
eeglab redraw;
