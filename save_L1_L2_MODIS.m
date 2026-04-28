clc;clear all
parentDir = 'F:\L1A_2021_2022\MODIS\L2O\';
load('F:\L1A_2021_2022\MODIS\dirname.mat')
secdir = 'F:\L1A_2021_2022\MODIS\secdata\';
% for idays = 1:length(dirname) %%%
for idays = 38:38 %%%    
    % G:\Multi_Sat_NN_AC\A_paper2\VIIRS_L1A_202001\L2S\20200101\00
    datestr = dirname{idays};
    % datestr = '20150801';
    ofdir=[parentDir,datestr,'\'];
    fdir = [ofdir];
    files=dir([fdir,'*.nc']); % NASA L2
    fdir2 = ['F:\L1A_2021_2022\MODIS\L2S\',datestr,'\'];

    for ii = 1:length(files)
    try
        fname=[fdir,files(ii).name];
        soloname = files(ii).name;
        longitude = ncread(fname,'/navigation_data/longitude');
        latitude = ncread(fname,'/navigation_data/latitude');
        year = ncread(fname,'/scan_line_attributes/year');
%         day = ncread(fname,'/scan_line_attributes/day');
        yyyy = str2double(soloname(12:15));
        mmmm = str2double(soloname(16:17));
        dddd = str2double(soloname(18:19));
        hours = str2double(soloname(21:22));
        minute = str2double(soloname(23:24)); 
        time = [repmat([yyyy mmmm dddd],length(year),1)];
        time(:,4) = hours;
        time(:,5) = minute;
        time(:,6) = 0;
        
        % 下面五个波段是一样的空间mask
        fname2 = [fdir2,soloname(1:25),'0.L2S.nc'];
        
        if exist(fname2)==2
        Rrs_443 = ncread(fname,'/geophysical_data/Rrs_443');
        %tic
        [mask33]= ~isnan(Rrs_443);% 
        %save(['F:\L1A_2021_2022\L2\matches2\',dirname1,'\mask33_',dirname1,'_',num2str(ii),'.mat'],'mask33')
        %toc
        [lox loy] = find(mask33);
        if length(loy)>1
        %这里有217个中心点在这张图是满足的。
        cor_time = time(loy,:);
        cor_lon = longitude(mask33);
        cor_lat = latitude(mask33);
        Rrs_412 = ncread(fname,'/geophysical_data/Rrs_412');
        Rrs_443 = ncread(fname,'/geophysical_data/Rrs_443');
        Rrs_469 = ncread(fname,'/geophysical_data/Rrs_469');
        Rrs_488 = ncread(fname,'/geophysical_data/Rrs_488');
        Rrs_531 = ncread(fname,'/geophysical_data/Rrs_531');
        Rrs_547 = ncread(fname,'/geophysical_data/Rrs_547');
        Rrs_555 = ncread(fname,'/geophysical_data/Rrs_555');
        Rrs_645 = ncread(fname,'/geophysical_data/Rrs_645');
        Rrs_667 = ncread(fname,'/geophysical_data/Rrs_667');
        Rrs_678 = ncread(fname,'/geophysical_data/Rrs_678');


        rhot_412 = ncread(fname2,'/geophysical_data/rhot_412');
        rhot_443 = ncread(fname2,'/geophysical_data/rhot_443');
        rhot_488 = ncread(fname2,'/geophysical_data/rhot_488');
        rhot_531 = ncread(fname2,'/geophysical_data/rhot_531');
        rhot_547 = ncread(fname2,'/geophysical_data/rhot_547');
        rhot_667 = ncread(fname2,'/geophysical_data/rhot_667');
        rhot_678 = ncread(fname2,'/geophysical_data/rhot_678');
        rhot_748 = ncread(fname2,'/geophysical_data/rhot_748');
        rhot_869 = ncread(fname2,'/geophysical_data/rhot_869');
        rhot_1240 = ncread(fname2,'/geophysical_data/rhot_1240');
        rhot_1640 = ncread(fname2,'/geophysical_data/rhot_1640');

        solz = ncread(fname2,'/geophysical_data/solz');
        senz = ncread(fname2,'/geophysical_data/senz');
        sena = ncread(fname2,'/geophysical_data/sena');
        sola = ncread(fname2,'/geophysical_data/sola');
        glint_coef =  ncread(fname2,'/geophysical_data/glint_coef');
        l2_flags = ncread(fname2,'/geophysical_data/l2_flags');
        % 1-6 time % 7-8 lonlat % 9-18 rhots % 19-23 Rrs
        % 24-27 angles
        rhots = [rhot_412(mask33),rhot_443(mask33),rhot_488(mask33),rhot_531(mask33),...
            rhot_547(mask33),rhot_667(mask33),rhot_678(mask33),rhot_748(mask33),...
            rhot_869(mask33),rhot_1240(mask33),rhot_1640(mask33)];
        Rrss = [Rrs_412(mask33),Rrs_443(mask33),Rrs_469(mask33),Rrs_488(mask33),...
            Rrs_531(mask33),Rrs_547(mask33),Rrs_555(mask33),Rrs_645(mask33),Rrs_667(mask33),Rrs_678(mask33)];

        viewing_angles = [solz(mask33),senz(mask33),sena(mask33),sola(mask33)];
        sec_data = [cor_time,cor_lon,cor_lat,rhots,Rrss,viewing_angles,double(l2_flags(mask33)),glint_coef(mask33)];
        
        disp(['已经匹配完的第',num2str(ii),'文件: ',soloname, ...
            ',共有',num2str(length(files)),'个文件  '])

        if ~exist([secdir,datestr,'\'])
            mkdir([secdir,datestr,'\'])
        else
            cc = 1;
        end


        save([secdir,datestr,'\','MODIS_',datestr,'T', ...
           soloname(12:26),'.mat'],'sec_data','-v7.3')
        else
        disp(['文件内无内容：已经匹配完的第',num2str(ii),'文件: ',soloname,',共有',num2str(length(ii)),'个文件  '])
        end
        end
    catch
        disp(['NC文件出错。第',num2str(ii),'文件: ',soloname,',共有',num2str(length(ii)),'个文件  ',datestr])
    end

    end


end


% date = datenum(2020,01,01): datenum(2020,01,31);
% date1 = repmat(date,1,24);
% date1 = sort(date1);
% date1 = datevec(date1);
% dayn = repmat(0:23,1,31);
% date1(:,4) = dayn;