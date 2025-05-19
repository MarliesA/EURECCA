clc;
clear;
close all;
warning('off', 'all');

%INVULSECTIE
plotSwaths = 1;
save_swaths = 1;
plotting = 1;      % plot final results?
% levelbed = 1;
saving = 1;

outputPathplot = '\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\movmean_footprint2\fig\';
outputPathdata = '\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\movmean_footprint2\data\';
campaign = 'SedRip';
frame = 'L4C1';
outputPathSwath = '\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\movmean_footprint2\swath\';

% location of the data
workPath = '\\tudelft.net\staff-umbrella\EURECCA\Floris\fieldwork\raw_data\texel\SRPLS\';
mapinfo = dir([workPath, '*2023']);
dates = size(mapinfo,1);

% instrument data, copy from metadata file
% Below campaign ripsed
SONAR3D(1).timeIN = datenum([2023,11,1,16,00,00]);     % net voor de eerste inzet
SONAR3D(1).timeOUT = datenum([2023,11,10,12,00,00]);   % net voor servicen
SONAR3D(1).zeroOrientation = 161.30; %111.30; %305.0;%267.1;                     % = offset x-axis with respect to AQD + angle between AQD and North
SONAR3D(1).z = 0.74;                                 % Bottom head center ==> not sampling volume yet!
SONAR3D(1).depth = 0.7;                                 % approximate depth
SONAR3D(1).location = [116104, 558946.6];               % RDx,RDy

% Below SEDMEX
% SONAR3D(1).timeIN = datenum([2021,09,10,00,00,00]);     % net voor de eerste inzet
% SONAR3D(1).timeOUT = datenum([2023,11,10,12,00,00]);   % net voor servicen
% SONAR3D(1).zeroOrientation = 111.30; %305.0;%267.1;                     % = offset x-axis with respect to AQD + angle between AQD and North
% SONAR3D(1).z = 0.74;                                 % Bottom head center ==> not sampling volume yet!
% SONAR3D(1).depth = 0.7;                                 % approximate depth
% SONAR3D(1).location = [116104, 558946.6];               % RDx,RDy
% 
% SONAR3D(2).timeIN = datenum([2021,11,1,16,00,00]);     % net na servicen
% SONAR3D(2).timeOUT = datenum([2023,11,10,12,00,00]);   % net na recovery
% SONAR3D(2).zeroOrientation = 111.30; %267.1;                   % = offset x-axis with respect to AQD + angle between AQD and North
% SONAR3D(2).z = SONAR3D(1).z;                          % Bottom head center ==> not sampling volume yet!
% SONAR3D(2).depth = SONAR3D(1).depth;                  % approximate depth
% SONAR3D(2).location = SONAR3D(1).location;            % RDx,RDy

smallscalefilter = 0.05;
%%Eind invulsectie

%%
tel = 0;
zmiddlePert_l = [];
zdates = [];
hwait = waitbar(0, 'Progress');
% read the data
for date = 1:dates(1);
    fileInfo = dir([workPath, mapinfo(date).name, '/', '*.RW2']);
    nRawFiles = size(fileInfo,1);
    for h = 1:nRawFiles
        tel = tel+1;
        fileName = fileInfo(h).name;
        fprintf(1,'Filename: %s\n',fileInfo(h).name);
        year = mapinfo(date).name(5:8);
        month = mapinfo(date).name(3:4);
        day = mapinfo(date).name(1:2);
        hour = fileName(1:2);
        min = fileName(3:4);
        timestamp = fileName(1:8);
        
%         if str2double(hour)<3
%             continue
%         end
%         if str2double(min)<30
%             continue
%         end

        % append file timestamp (name) to measurement 
        zdates = [zdates; string(timestamp)];
        [header, data, range] = readSonar3DRW2_mac([workPath,mapinfo(date).name,'/',fileName]);
        if datenum(header.whenStarted)<SONAR3D(1).timeIN||datenum(header.whenStarted)>SONAR3D(end).timeOUT
            continue
        end
        
        timevec(tel,:) = header.whenStarted;
        % The header contains relevant information on how the 3D Sonar was
        % programmed, for example, the arcwidth centered around the vertical axis,
        % the swathstep, and rotationstep. Part of the information within the
        % header has already been used to compute the range.
        
        % The data are the signal amplitudes organized as [Nsamples Npings
        % NSwaths]. That is, a single swath contains Npings (profiles), each from
        % range(1) to range(end), with each profile comprising Nsamples. In total
        % NSwaths were performed to cover a full circle. Nsamples equals the length
        % of range, Npings equals header.arc.
        [Nsamples, Npings, Nswaths] = size(data);
        
        % The unit stepsize in a swath and a rotation step is 1 in the 400-degree
        % system
        stepSize = 1*360/400;
        
        % The processing is carried out in the following steps:
        % (1) Determine bed level for each swath
        % (2) Put into (x,y,z)
        % (3) Rotate to have y north and x east
        % (4) Outlier detection, probably redundant
        % (5) Interpolate to a regular grid, and remove trend to reveal ripples as
        %     perturbations
        
        
        %% STEP 1
        
        % We are going to make a "virtual" x-z axis for each swath, irrespective of
        % the rotation angle. This is going to result in a bed profile from each
        % swath. Unfortunately, the manual is not very specific about the angles
        % within the swath, other than that the arc is centered around the
        % vertical. So, 0 should be pointing straight down. The Npings is even,
        % which suggests that there are Npings/2 < 0 and Npings/2 > 0. Curiously,
        % for the Bardex II data, the viewer that comes with the sonar produces a
        % marked difference between the first and last swath in terms of bed
        % elevation. This is not due to some overall roll or tilt, but indicates
        % 'uncertainties' with how the angles within the arc are defined. This
        % difference between first and last swath is not found for occasions when
        % the sonar is externally triggered. In Bardex II, it was triggered on-line
        % from a laptop. Could that matter?
        
        % Interestingly, we do not know whether the Sonar3D samples from negative
        % to positive angles, or from positive to negative. We will assume here
        % that it does so from negative to positive. In the swath with rotation =
        % 0, the positive x goes out from the 0 rotation angle of the sonar head.
        % Or, said for Bardex II, the 0 pointed approximately to wave paddle. So,
        % positive x for rotation = 0 is pointing toward the wave paddle and
        % negative x to the swash zone.
        
        slotoffset = 0;
        THdeg = (-header.Arc/2:header.SwathStep:header.Arc/2-1).*stepSize + slotoffset*stepSize;
        THrad = deg2rad(THdeg);
        
        % Note: for externally triggered systems it appears that slotoffset should be 0.
        % And curiously, some Bardex II scans required step to be -2. For AZG frame
        % 5, slotoffset was -1. For DVA1, several values were tested and 0 is correct.
        
        % Turn range and TH into Cartesian coordinates. This is a coordinate scheme
        % within this swath, with X the horizontal component and Z the vertical
        % component.
        THETA = ones(length(range),1)*THrad;
        RHO = ones(length(THrad),1)*range;
        [Zswath,Xswath] = pol2cart(THETA',RHO);       % Convert swaths from polar to cartesian
        
        
        % Put all xBed and zBed into a proper (x,y,z) matrix
%       dir0 = 90-SONAR3D(1).zeroOrientation -30  % in cartesian coords
        dir0 = 270-SONAR3D(1).zeroOrientation - 10;  % in cartesian coords
        beachOri = 123.4; %146.6; 
        THrot = mod((0:header.RotateStep:200-header.RotateStep)*stepSize - dir0 -beachOri + 360, 360);
        THrotrad = deg2rad(THrot);
        
        % Note Zswath and Xswath are now Npings x Nsamples, which is different from the order
        % in data. The bed level detection algorithm called below assumes Npings x
        % Nsamples. There is a transpose there for each swath in data.
        %
        % And, pol2cart has Zs,Xs as output as THETA is defined with respect to the
        % x-axis, which is here vertically down, and hence Z.
        
        % change sign of Zswath to have below transducer head as negative
        Zswath = -Zswath;
        
        % Prepare output. For every ping we are going to find a bed level at (xBed,zBed)
        xBed = NaN(Npings,Nswaths);
        zBed = xBed;
       
        
        % The bed is expected between minZ and belowBedZ. The signal beneath belowBedZ is used to quantify noise.
        % At least for the UU frame, the height of the Sonar was ~0.9 m, so we take
        % minZ as sensorheight/2 and belowBedZ as sensorheight*2. However,
        % if the
        % bed level changes are large, this might need to be changed!
        minZ = -0.6;
        belowBedZ = -1.1;
        disp('Starting swaths')
        for i = 1:Nswaths          % for every swath
            
            bS = data(:,:,i)';     % input signal amplitudes, with transpose to go Npings x Nsamples
            [xBed(:,i), zBed(:,i)] = getBedLevelFromSingleSwath(Xswath,Zswath,bS,minZ,belowBedZ);  % get bed level estimates for this swath
            
            if plotSwaths   % if 1, show intermediate results
                names = linspace(1, Nswaths);
                figure('visible', 'off');
%                 figure
                pcolor(Xswath,Zswath,bS);
                shading flat;
                hold on;
                plot(xBed(:,i),zBed(:,i),'ok','markerSize',2);
                hold off
                leg.Interpreter="latex";
                ylabel('elevation (from sonar head level) [m]')
                xlabel('x [m]')
                title(['Swath ' num2str(i) ' of ' num2str(Nswaths)])
%                 pause(0.1);
                if save_swaths
                    saveas(gca, fullfile(outputPathSwath, ['202411', day, fileName(1:end-8), '_', replace(num2str(THrot(i)),'.','_'), '_ithet', num2str(i), '.png']))
%                     databed.x = xBed(:,i);
%                     databed.z = zBed(:,i);
%                     save([outputPathSwath, num2str(h), '_', num2str(i), '.mat'],'databed');
                end 
                close all
            end
%             close all 
            
        end
        disp('end swaths')
        % break
        
%         figure();
%         plot(xBed,zBed)
%         xlabel('x (m)')
%         ylabel('z (m)')
        %% STEP 2
        

        
%         [mini, iarg] = min(abs(THrot-146.6))
%         scatter(X(:, iarg), Y(:, iarg), 'c')
        
        THETArot = ones(Npings,1)*THrotrad;
        [X,Y] = pol2cart(THETArot,xBed);
        
        % Turn these into columns and remove NaNs
        XCol = X(:);
        YCol = Y(:);
        ZCol = zBed(:);
%         ZCol(isnan(Zcol))= -10;
        nandat = isnan(XCol)|isnan(YCol)|isnan(ZCol);
        XCol(nandat) = [];
        YCol(nandat) = [];
        ZCol(nandat) = [];
        
        if length(ZCol)<10
            fprintf('only 10 valid bed points, therefore continue')
            continue
        end
%         
%         figure();
%         scatter(YCol,XCol,2,ZCol);
%         xlabel('x (m)')
%         ylabel('y (m)')
%         title('Depth below sensor')
%         hcb=colorbar;
%         title(hcb,'(m)')
%         axis equal
%         hold on
%         scatter(X(:, 49), Y(:,49), 'g', 'filled')
%         scatter(X(:, 24), Y(:,24), 'c', 'filled')
%         scatter(X(:, 99), Y(:,99), 'r', 'filled')
        %% STEP 5
        
        % Interpolate to a regular grid, from  in both x and y with a
        % 0.01 m step size. Use lx and ly to fill gaps; note that this larger than
        % for the perturbations from the central line. 

        xGrid = -0.9:0.01:0.9;
        yGrid = xGrid;
        
        lxs = 0.05;   % small
        lys = 0.05;
        [xi,yi,zGrids,eGrids] = loess_grid2dh(XCol,YCol,ZCol,xGrid,yGrid,lxs,lys);
        zGrids(eGrids > 0.02) = NaN;  % remove points at the border
        zGrids = zGrids';
        
        
        %% Plotting
        if plotting
            figure('visible','off');
            pcolor(xi, yi, zGrids); shading('flat');               
            hcb=colorbar;
            title(hcb,'[m]')
            hold on
            xlabel('x (m)');
            ylabel('y (m)');
            axis square
            xlim([-0.9, 0.9])
            clim([-0.95 -0.75])
            colormap('jet')
            runn = [day '-' month '-' year ' ' hour ':' min];
            title(runn)
            print([outputPathplot,campaign,'_',year,month,day,hour,min,'.png'],'-dpng','-r300');
            close all
        end
        
        if saving
        %% store data
            data05.x = xi;
            data05.y = yi;
            data05.xBed = xBed;
            data05.zBed = zBed;
            data05.THrot = THrot;
            data05.z = zGrids;
            data05.e = eGrids;
            data05.header = header;
            data05.SONARdata = SONAR3D;
            data05.flag = 1;
            save([outputPathdata, [fileName(1:4) mapinfo(date).name],'.mat'],'data05');
   
        end
        waitbar(h/nRawFiles, hwait, sprintf('Progress: %d%%', round(h/nRawFiles * 100)));
    end
end
close(hwait)
