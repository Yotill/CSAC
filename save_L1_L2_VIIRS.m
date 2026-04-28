clc;clear all
parentDir = 'F:\L1A_2021_2022\VIIRS\L2S\';
fileList = dir(parentDir);
subFolders = fileList([fileList.isdir] & ~ismember({fileList.name}, {'.', '..'}));
dirname = {subFolders.name}';
secdir = 'F:\L1A_2021_2022\VIIRS\secdata_rhot\';
% for idays = 1:length(dirname) %%%
for idays = 46:60 %%%    
    % G:\Multi_Sat_NN_AC\A_paper2\VIIRS_L1A_202001\L2S\20200101\00
    datestr = dirname{idays};
    % datestr = '20150801';
    ofdir=['F:\L1A_2021_2022\VIIRS\L2S\',datestr,'\'];
    fdir = [ofdir];
    files=dir([fdir,'*.nc']); % NASA L2
    fdir2 = ['F:\L1A_2021_2022\L2\',datestr,'\'];
    % for ii = 1:length(files)
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
        
        fname2 = [fdir2,soloname(1:29),'.OC.nc'];
        if exist(fname2)==2
        Rrs_443 = ncread(fname2,'/geophysical_data/Rrs_443');
        %tic
        % [mask33]= ~isnan(Rrs_443);% 这里的mask可能需要改变？？？
        rhot_443 = ncread(fname,'/geophysical_data/rhot_443');
        % mask33 需要考虑flags 01-04-08-09
        l2_flags = ncread(fname,'/geophysical_data/l2_flags');
        flags = dec2bin(l2_flags);
        f1 =  double(flags(:,end-1)); f2 = double(flags(:,end-4));
        f3 =  double(flags(:,end-8)); f4 = double(flags(:,end-9));
        outflags = f1==0&f2==0&f3==0&f4==0;
        out1 = zeros(size(rhot_443)); out1(:) = outflags;
        mask33 = (abs(rhot_443)<0.3|~isnan(Rrs_443))&rhot_443>0.01&out1==0;
        %save(['F:\L1A_2021_2022\L2\matches2\',dirname1,'\mask33_',dirname1,'_',num2str(ii),'.mat'],'mask33')
        %toc
        [lox loy] = find(mask33);
        if length(loy)>1
        %这里有217个中心点在这张图是满足的。
        cor_time = time(loy,:);
        cor_lon = longitude(mask33);
        cor_lat = latitude(mask33);
        Rrs_410 = ncread(fname2,'/geophysical_data/Rrs_410');
        Rrs_486 = ncread(fname2,'/geophysical_data/Rrs_486');
        Rrs_551 = ncread(fname2,'/geophysical_data/Rrs_551');
        Rrs_671 = ncread(fname2,'/geophysical_data/Rrs_671');

        rhot_410 = ncread(fname,'/geophysical_data/rhot_410');
        rhot_443 = ncread(fname,'/geophysical_data/rhot_443');
        rhot_486 = ncread(fname,'/geophysical_data/rhot_486');
        rhot_551 = ncread(fname,'/geophysical_data/rhot_551');
        rhot_671 = ncread(fname,'/geophysical_data/rhot_671');
        rhot_745 = ncread(fname,'/geophysical_data/rhot_745');
        rhot_862 = ncread(fname,'/geophysical_data/rhot_862');
        rhot_1238 = ncread(fname,'/geophysical_data/rhot_1238');
        rhot_1601 = ncread(fname,'/geophysical_data/rhot_1601');
        rhot_2257 = ncread(fname,'/geophysical_data/rhot_2257');

        solz = ncread(fname,'/geophysical_data/solz');
        senz = ncread(fname,'/geophysical_data/senz');
        sena = ncread(fname,'/geophysical_data/sena');
        sola = ncread(fname,'/geophysical_data/sola');
        glint_coef = ncread(fname,'/geophysical_data/glint_coef');
        l2_flags = ncread(fname,'/geophysical_data/l2_flags');

        
        % 1-6 time % 7-8 lonlat % 9-18 rhots % 19-23 Rrs
        % 24-27 angles
        rhots = [rhot_410(mask33),rhot_443(mask33),rhot_486(mask33),rhot_551(mask33),...
            rhot_671(mask33),rhot_745(mask33),rhot_862(mask33),rhot_1238(mask33),...
            rhot_1601(mask33),rhot_2257(mask33)];
        Rrss = [Rrs_410(mask33),Rrs_443(mask33),Rrs_486(mask33),Rrs_551(mask33),Rrs_671(mask33)];
        viewing_angles = [solz(mask33),senz(mask33),sena(mask33),sola(mask33),glint_coef(mask33)];
        sec_data = [cor_time,cor_lon,cor_lat,rhots,Rrss,viewing_angles,double(l2_flags(mask33))];
        
        disp(['已经匹配完的第',num2str(ii),'文件: ',soloname, ...
            ',共有',num2str(length(files)),'个文件  '])

        if ~exist([secdir,datestr,'\'])
            mkdir([secdir,datestr,'\'])
        else
            cc = 1;
        end


        save([secdir,datestr,'\','VIIRS_',datestr,'T', ...
           soloname(12:26),'.mat'],'sec_data','mask33','-v7.3')
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