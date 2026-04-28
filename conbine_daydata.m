clc;clear all
L2_dir = 'F:\L1A_2021_2022\MODIS\L2O\';
load('F:\L1A_2021_2022\MODIS\dirname.mat')
for idays = 1:60 %%%
    % G:\Multi_Sat_NN_AC\A_paper2\VIIRS_L1A_202001\L2S\20200101\00
    datestr = dirname{idays};
    fdir=['F:\L1A_2021_2022\MODIS\secdata\secdata_chl\',datestr,'\'];
    % 可以加个flag0
    files=dir([fdir,'*.mat']); 
    alldata = [];
    for ii = 1:length(files)
        fname=[fdir,files(ii).name];
        load(fname)
        % alldata = [alldata;all_data];
        alldata = [alldata;sec_data];
        disp(['已经匹配完的第',num2str(ii),'文件: ',datestr, ...
            ',共有',num2str(length(files)),'个文件  '])
    end
    save(['F:\L1A_2021_2022\MODIS\daydata_l1l2\daydata_chl\', ...
        'modis_',datestr,'.mat'],'alldata','-v7.3')

end


% date = datenum(2020,01,01): datenum(2020,01,31);
% date1 = repmat(date,1,24);
% date1 = sort(date1);
% date1 = datevec(date1);
% dayn = repmat(0:23,1,31);
% date1(:,4) = dayn;