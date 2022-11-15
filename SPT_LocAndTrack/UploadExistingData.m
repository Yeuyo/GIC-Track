%   SerialProcess_PALM_JF549.m
%   Anders Sejr Hansen, August 2016
%   Yew Yan Wong, March 2022
clear; clc; close all; clearvars -global

%   DESCRIPTION
%   This script takes as input a folder with tiff files and then outputs
%   workspaces with tracked single molecules. Briefly, it uses the
%   BioFormats package to read in nd2 files. Next, the script feeds the
%   images as a 3D matrix into the localization part of the MTT algorithm
%   (Part 1) and subsequently, the tracked particles are fed into the
%   tracking part of the MTT algorithm (Part 2). 

%%%%%%%%%%%%%%%%%%%% DEFINE INPUT AND OUTPUT PATHS %%%%%%%%%%%%%%%%%%%%%%%%
% add the neccesary paths:
addpath(genpath(['.' filesep 'Batch_MTT_code' filesep])); % MTT & BioFormats
addpath(genpath(['.' filesep 'CMP_Fit' filesep])); % Component fittings
disp('added paths for MTT algorithm mechanics, bioformats...');

load("../tifupload.mat");

if iscell(file_names)
  Filenames = file_names;
else
  Filenames = cell(1,1);
  Filenames{1} = file_names;
end
% Check if the user's machine has the number of cores they desired to use
if numCores > feature('numcores')
  %   reduce the cores to use to the machine's maximum number of cores
  numCores = feature('numcores');
end
if numCores == 1 || length(Filenames) == 1
  runParallel = false;
end

%% Producing diffusion data
tracked_files = cell(numel(Filenames), 1);
for n = 1 : numel(Filenames)
  tracked_files{n} = Filenames{n};
end
subPlotNumbers = ceil(sqrt(length(tracked_files)));

figure;
for iter = 1:length(tracked_files)
  maxJump = 0;
  
  load([input_path, tracked_files{iter}]);
  
  % trackedPar to tracks
  tracks = struct('data', []);
  for n = 1 : numel(trackedPar)
    tracks(n) = struct('data', [reshape(trackedPar(n).TimeStamp, [], 1), trackedPar(n).xy]);
  end
  
  rows_to_remove = [];
  for n = 1 : numel(tracks)
    if size(tracks(n).data, 1) < traj_length
      rows_to_remove = [rows_to_remove; n];
    end
  end
  tracks(rows_to_remove) = [];
  
  msd = computeMSD(tracks, tol);
  lfit = fitMSD(msd, clip_factor, analysis_type);
  
  subplot(subPlotNumbers, subPlotNumbers, iter);
  hist(log10(lfit.a(lfit.a>0)), 15);
  title(Filenames{iter}, 'interpreter', 'none');
  xlabel('Log10(D(um2/s)', 'interpreter', 'latex');
  ylabel('Counts', 'interpreter', 'latex');
  
  D = log10(lfit.a(lfit.a>0));
  E = lfit.a(lfit.a>0);

  tracks = tracks(lfit.a>0);
  msd = msd(lfit.a>0);
  dataTraj = [];
  dataTrack = [];
  for n = 1 : numel(tracks)
    traj_size = size(tracks(n).data, 1);
    dataTraj = [dataTraj; convertCharsToStrings(Filenames{iter}), n, traj_size, mean(msd{n}(1 : traj_size, 2)), D(n), tracks(n).data(1,1), tracks(n).data(end,1)];
    dataTrack = [dataTrack; repelem(n, traj_size)', tracks(n).data, msd{n}(1 : traj_size, 2)];
  end
  dataTraj = cellstr(dataTraj);
  save([output_path, Filenames{iter}, '_dataTraj.mat'], 'dataTraj');
  save([output_path, Filenames{iter}, '_dataTrack.mat'], 'dataTrack');
  
%   save([output_path, Filenames{iter}, '_Log_Diffusion.mat'], 'D');
  save([output_path, Filenames{iter}, '_Diffusion.mat'], 'E');
  save([output_path, Filenames{iter}, '_msd.mat'], 'msd');

  % Jump distance computation
  trackedPar(rows_to_remove) = [];
  trackedPar = trackedPar(lfit.a>0);
  jdTracks = [];
  for n = 1 : numel(trackedPar)
%     jdTracks = [jdTracks; tracks(n).xy, tracks(n).Frame, repelem(n, length(tracks(n).Frame))', repelem(D(n), length(tracks(n).Frame))'];
    jdTracks = [jdTracks; trackedPar(n).xy, reshape(trackedPar(n).Frame, [], 1), repelem(n, length(trackedPar(n).Frame))', repelem(D(n), length(trackedPar(n).Frame))'];
    if length(trackedPar(n).Frame) > maxJump
      maxJump = length(trackedPar(n).Frame);
    end
  end
  binEdges = 0 : ExposureTime / 1000 : maxJump;
  [tlist, rlist, JDH] = calculateJDH(jdTracks, 1, binEdges, ExposureTime / 1000, clip_factor, 0, 0);
  % Isolate first histogram of displacements;
  FJH = [];
  FJH(:,1) = rlist;
  FJH(:,2) = JDH(1,:); % only maxFrame = 1 is used
  D0 = 1;

  % Fit first histogram of displacements;
  [FJH_Coef1Cmp,FJH_Fit1Cmp,FJH_Sigma1Cmp] = JDHfixedT_1cmp_fit(FJH, ExposureTime / 1000, D0);
  [FJH_Coef2Cmp,FJH_Fit2Cmp, FJH_Sigma2Cmp] = JDHfixedT_2cmp_fit(FJH, ExposureTime / 1000, [10*D0, D0]);
  [FJH_Coef3Cmp,FJH_Fit3Cmp, FJH_Sigma3Cmp] = JDHfixedT_3cmp_fit(FJH, ExposureTime / 1000, [10*D0, D0, 0.1*D0]);

  % Add fits of first histogram of displeacements to FJH.
  FJH(:,3) = FJH_Fit1Cmp(:,2);
  FJH(:,4) = FJH_Fit2Cmp(:,2);
  FJH(:,5) = FJH_Fit3Cmp(:,2);

  rlist(1,2) =tlist(1);

  FitPar = [FJH_Coef1Cmp, FJH_Coef2Cmp, FJH_Coef3Cmp;...
    FJH_Sigma1Cmp, 0, FJH_Sigma2Cmp, 0, FJH_Sigma3Cmp, 0];

  % Two component diffusion fit
  FirstCMP = ...
    JDHfixedT_1cmp_fun([FitPar(1,4)* FitPar(1,7), FitPar(1,5)],rlist);
  SecondCMP = ...
    JDHfixedT_1cmp_fun([FitPar(1,4)* (1 - FitPar(1,7)), FitPar(1,6)],rlist);
  twoCMPFit = [FirstCMP, SecondCMP];

  % Three component diffusion fit
  FirstCMP = ...
    JDHfixedT_1cmp_fun([FitPar(1,9)* FitPar(1,13), FitPar(1,10)],rlist);
  SecondCMP = ...
    JDHfixedT_1cmp_fun([FitPar(1,9)* FitPar(1,14), FitPar(1,11)],rlist);
  ThirdCMP = JDHfixedT_1cmp_fun...
    ([FitPar(1,9)* (1 - FitPar(1,13) - FitPar(1,14)), FitPar(1,12)],rlist);
  threeCMPFit = [FirstCMP, SecondCMP, ThirdCMP];

  save([output_path, Filenames{iter}, '_CMPFitPar.mat'], 'FJH');
  save([output_path, Filenames{iter}, '_FitPar.mat'], 'FitPar');
  save([output_path, Filenames{iter}, '_2CMPFit.mat'], 'twoCMPFit');
  save([output_path, Filenames{iter}, '_3CMPFit.mat'], 'threeCMPFit');
end
sgtitle('Diffusion Coefficient for all population', 'interpreter', 'latex');

clear clip_factor analysis_type tracks rows_to_remove lfit D E msd ...
      subPlotNumbers tracked_files trackedPar traj_length track_length ...
      D0 tlist rlist JDH jdTracks binEdges ...
      FJH_Coef1Cmp FJH_Fit1Cmp FJH_Sigma1Cmp ...
      FJH_Coef2Cmp FJH_Fit2Cmp FJH_Sigma2Cmp ...
      FJH_Coef3Cmp FJH_Fit3Cmp FJH_Sigma3Cmp ...
      FitPar FirstCMP SecondCMP ThirdCMP twoCMPFit threeCMPFit

%% Functions
function msd = computeMSD(tracks, tol)
indices = 1: numel(tracks);
n_tracks = numel(indices);
all_delays = cell(n_tracks, 1);
for i = 1 : n_tracks
  index = indices(i);
  track = tracks(index).data;
  t = track(:,1);
  [T1, T2] = meshgrid(t, t);
  dT = round(abs(T1(:)-T2(:)), tol);
  all_delays{i} = unique(dT);
end
delays = unique( vertcat(all_delays{:}) );
n_delays = numel(delays);
msd = cell(n_tracks, 1);
for i = 1 : n_tracks
  mean_msd    = zeros(n_delays, 1);
  M2_msd2     = zeros(n_delays, 1);
  n_msd       = zeros(n_delays, 1);
  
  index = indices(i);
  track = tracks(index).data;
  t = track(:,1);
  t = round(t, tol);
  X = track(:, 2:end);
  
  n_detections = size(X, 1);
  
  for j = 1 : n_detections - 1
    
    % Delay in physical units
    dt = t(j+1:end) - t(j);
    dt = round(dt, tol);
    
    % Determine target delay index in bulk
    [~, index_in_all_delays, ~] = intersect(delays, dt);
    
    % Square displacement in bulk
    dX = X(j+1:end,:) - repmat(X(j,:), [(n_detections-j) 1] );
    dr2 = sum( dX .* dX, 2);
    
    % Store for mean computation / Knuth
    n_msd(index_in_all_delays)     = n_msd(index_in_all_delays) + 1;
    delta = dr2 - mean_msd(index_in_all_delays);
    mean_msd(index_in_all_delays) = mean_msd(index_in_all_delays) + delta ./ n_msd(index_in_all_delays);
    M2_msd2(index_in_all_delays)  = M2_msd2(index_in_all_delays) + delta .* (dr2 - mean_msd(index_in_all_delays));
  end
  
  n_msd(1) = n_detections;
  std_msd = sqrt( M2_msd2 ./ n_msd ) ;
  
  delay_not_present = n_msd == 0;
  mean_msd( delay_not_present ) = NaN;
  
  msd{index} = [ delays mean_msd std_msd n_msd ];
end
end

function lfit = fitMSD(msd, clip_factor, analysis_type)
if nargin < 2
  clip_factor = 0.25;
  analysis_type = "percentage";
end

n_spots = numel(msd);
if analysis_type == "percentage"
  if clip_factor < 1
    fprintf('Fitting %d curves of MSD = f(t), taking only the first %d%% of each curve... ',...
      n_spots, ceil(100 * clip_factor) )
  else
    fprintf('Fitting %d curves of MSD = f(t), taking only the first %d points of each curve... ',...
      n_spots, round(clip_factor) )
  end
end


a = NaN(n_spots, 1);
a_bounds = NaN(n_spots, 2);
b = NaN(n_spots, 1);
b_bounds = NaN(n_spots, 2);
r2fit = NaN(n_spots, 1);
ft = fittype('poly1');

fprintf('%4d/%4d', 0, n_spots);
for i_spot = 1 : n_spots
  
  msd_spot = msd{i_spot};
  
  t = msd_spot(:,1);
  y = msd_spot(:,2);
  w = msd_spot(:,4);
  
  if analysis_type == "percentage"
    % Clip data, never take the first one dt = 0
    if clip_factor < 1
      t_limit = 2 : round(numel(t) * clip_factor);
    else
      t_limit = 2 : min(1+round(clip_factor), numel(t));
    end
  elseif analysis_type == "number"
    t_limit = 2: clip_factor;
  end
  
  t = t(t_limit);
  y = y(t_limit);
  w = w(t_limit);
  
  % Thrash bad data
  nonnan = ~isnan(y);
  x = t(nonnan);
  y = y(nonnan);
  w = w(nonnan);
  
  if numel(y) < 2
    continue
  end
  
  [fo, gof] = fit(x, y, ft, 'Weights', w);
  
  if numel(x) > 3
    fo_bounds = confint(fo, 0.95);
    a_bounds(i_spot, :) = fo_bounds(1, :);
    b_bounds(i_spot, :) = fo_bounds(2, :);
  end
  
  a(i_spot) = fo.p1;
  b(i_spot) = fo.p2;
  r2fit(i_spot) = gof.adjrsquare;
  
end
lfit = struct(...
  'a', a, ...
  'a_bounds', a_bounds, ...
  'b', b, ...
  'b_bounds', b_bounds, ...
  'r2fit', r2fit);
end