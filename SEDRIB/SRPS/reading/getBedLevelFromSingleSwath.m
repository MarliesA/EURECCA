function [xBed,zBed] = getBedLevelFromSingleSwath(Xswath,Zswath,bS,minZ,belowBedZ)

% GETBEDLEVELFROMSINGLESWATH(Xswath,Zswath,BS,BEGINID,NOISEIDS) determines the bed
% level line from a single swath of the 3D Sonar.
%
% INPUT
%   Xswath, [nPings nSamples] matrix of x values
%   Zswath, [nPings nSamples] matrix of z values
%   bS, [nPings nSamples] matrix of backscatter
%   minZ, bins with a Zswath > minZ are not used. Make sure that minZ is not too close
%         to the actual bed.
%   belowBedZ, bins with a Zswath < belowBedZ are considered to be below bed.
%         Used for noise floor. Make sure belowBedZ is not too close to the
%         actual bed.
%
% OUTPUT
%   xBed, x-values of bed level line
%   zBed, z-values of bed level line
%      When zBed is a NaN, xBed is a dummy value! When interpoloating to a
%      regular grid afterwards, these dummy x ensure that gaps remain gaps.
%
% During the estimation of xBed and zBed, the algorithm will first detect the
% maximum backscatter with corresponding xBed and zBed. A second-order polynomial
% is then fitted centered around the maximum to go subgrid. This fit is based on 11 or more points.
%
% v1, 24 March 2015, Gerben Ruessink, modified from getBedLevelFromSRPSScan

% number of pings and samples
[nPings,nSamples] = size(Xswath);

% intialize output
xBed = NaN(nPings,1);
zBed = NaN(nPings,1);

bS(Zswath>=minZ) = 0;
bS2 = (movmean(double(bS>=250)', 15))';
 
% loop through each ping
for ping = 1:nPings
    mx = max(bS2(ping,:));
    I  = find(bS2(ping,:)==mx, 1, 'last');
    if I<800
        xBed(ping) = Xswath(ping, I);
        zBed(ping) = Zswath(ping, I);    
    else
        xBed(ping) = nan;
        zBed(ping) = nan;     
    end
%     % ids of interest
%     beginId = find(Zswath(ping,:)<minZ);
%     bS(:, 1:beginId) = 0;
%     idInt = beginId(1):nSamples;
    
    
    
%     if sum(bS(ping,idInt))~= 0  % there is a signal
%         [~, I] = max(bS(ping,:));
%         if I < 600
%             if sum(bS(ping, I:I+14))/15 == 255
%                 xBed(ping) = Xswath(ping, I);
%                 zBed(ping) = Zswath(ping, I);
%             else
%                 %disp('Using else')
%                 av = sum(bS(ping, I:I+14))/15;
%                 maxIter = 100;
%                 iter = 1;
%                 while av < 254.5 && iter < maxIter
%                     I = I+1;
%                     iter = iter + 1;
%                     av = sum(bS(ping, I:I+15))/16;
%                     %display(av)
%                 end
%                 if iter < 100
%                     xBed(ping) = Xswath(ping, I);
%                     zBed(ping) = Zswath(ping, I);
%                 end
%             end
%         end

%     end
    
end

dzs = NaN(length(zBed), 2);
for i=1:length(zBed)
    if i == 1
        dzs(i, 2) = abs(zBed(i) - zBed(i+1));
    elseif i == length(zBed)
        dzs(length(zBed), 1) = abs(zBed(i) - zBed(i-1));
    else
        dzs(i, 1) = abs(zBed(i) - zBed(i-1));
        dzs(i, 2) = abs(zBed(i) - zBed(i+1));
    end
end
for j = 1:length(zBed)
    if dzs(j, 1) > 0.02 && dzs(j, 2) > 0.02
        zBed(j) = NaN;
    end
    if  abs(zBed(j) - mean(zBed(~isnan(zBed)))) > 0.08
        zBed(j) = NaN;
    end
end


% detect and remove outliers
% Zcrit = 2.5;   % trial and error based value
% while 1
%     zNoTrend = zBed;
%     zNoTrend(~isnan(zBed)) = poly_detrend(xBed(~isnan(zBed)),zBed(~isnan(zBed)),2) + mean(zBed);
%     Z = (zNoTrend - mean(zNoTrend)) / std(zNoTrend);
%     if any(abs(Z) > Zcrit)
%         zBed(abs(Z) > Zcrit) = NaN;
%     else
%         break;
%     end
% end

% figure
% pcolor(Xswath, Zswath, bS)
% shading flat
% hold on
% scatter(xBed, zBed)
return
