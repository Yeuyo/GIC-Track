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
bins = 20; % temp, will move plotting to dashboard

%%%%%%%%%%%%%% READ IN TIF FILES %%%%%%%%%%%%%%%%
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
tWhole = tic;
if runParallel
  temp_struct_for_save = struct;
  % if there's less files to run than number of cores, reduce the core to use even further
  parpool('local', min([numCores, length(Filenames)]));
  parfor iter = 1:length(Filenames)
    disp('-----------------------------------------------------------------');
    tic;
    disp(['reading in TIFF file ', num2str(iter), ' of ', num2str(length(Filenames)), ' total TIFF files']);
    %%% read tiff files:
    [stack, nbImages] = tiffread([input_path, Filenames{iter}, '.tif']);
    
    % convert to 3D matrix as double:
    imgs_3d_double = zeros(size(stack(1,1).data,1), size(stack(1,1).data,2), nbImages);
    for img_iter = 1:nbImages
      imgs_3d_double(:,:,img_iter) = stack(img_iter).data;
    end
    
    toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%% MTT ALGORITHM PART 1: LOCALIZE ALL PARTICLES %%%%%%%%%%%%%%
    disp('MTT ALGORITHM PART 1: localize particles in all of the workspaces');
    tic;
    disp(['localizing all particles in movie number ', num2str(iter), ' of ', num2str(length(Filenames))]);
%     impars.name = Filenames{iter};
    data = localizeParticles_ASH(input_path, impars, locpars, imgs_3d_double);
    toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%% MTT ALGORITHM PART 2: TRACK PARTICLES BETWEEN FRAMES %%%%%%%%%
    disp('MTT ALGORITHM PART 2: track particles between frames');
    tic;
    disp(['tracking all localized particles from movie ', num2str(iter), ' of ', num2str(length(Filenames))]);
    data=buildTracks2_ASH(output_path_further_processing, Filenames{iter}, data, impars, locpars, trackpars, data.ctrsN, imgs_3d_double);
    toc;
    
    temp_struct_for_save(iter).data_cell_array = data.tr;
    temp_struct_for_save(iter).Width = size(imgs_3d_double,2);
    temp_struct_for_save(iter).Height = size(imgs_3d_double,1);
    temp_struct_for_save(iter).Frames = size(imgs_3d_double,3);
  end
else % single core
  for iter = 1:length(Filenames)
    disp('-----------------------------------------------------------------');
    tic;
    disp(['reading in TIFF file ', num2str(iter), ' of ', num2str(length(Filenames)), ' total TIFF files']);
    %%% read tiff files:
    [stack, nbImages] = tiffread([input_path, Filenames{iter}, '.tif']);
    
    % convert to 3D matrix as double:
    imgs_3d_double = zeros(size(stack(1,1).data,1), size(stack(1,1).data,2), nbImages);
    for img_iter = 1:nbImages
      imgs_3d_double(:,:,img_iter) = stack(img_iter).data;
    end
    
    toc;
    clear stack
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%% MTT ALGORITHM PART 1: LOCALIZE ALL PARTICLES %%%%%%%%%%%%%%
    disp('MTT ALGORITHM PART 1: localize particles in all of the workspaces');
    tic;
    disp(['localizing all particles in movie number ', num2str(iter), ' of ', num2str(length(Filenames))]);
    impars.name = Filenames{iter};
    data = localizeParticles_ASH(input_path, impars, locpars, imgs_3d_double);
    toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%% MTT ALGORITHM PART 2: TRACK PARTICLES BETWEEN FRAMES %%%%%%%%%
    disp('MTT ALGORITHM PART 2: track particles between frames');
    tic;
    disp(['tracking all localized particles from movie ', num2str(iter), ' of ', num2str(length(Filenames))]);
    data=buildTracks2_ASH(output_path_further_processing, Filenames{1}, data, impars, locpars, trackpars, data.ctrsN, imgs_3d_double);
    toc;
    
    %%%%%%%% SAVE THE TRAJECTORIES TO YOUR STRUCTURED ARRAY FORMAT %%%%%%%%
    tic;
    disp(['saving MATLAB workspace for movie ', num2str(iter), ' of ', num2str(length(Filenames))]);
    data_cell_array = data.tr;
    if length(data_cell_array) < min_traj
      continue
    end
    % save meta-data
    settings.Delay = impars.FrameRate;
    settings.px2micron = impars.PixelSize;
    settings.TrackingOptions = trackpars;
    settings.LocOptions = locpars;
    settings.AcquisitionOptions = impars;
    settings.Filename = impars.name;
    settings.Width = size(imgs_3d_double,2);
    settings.Height = size(imgs_3d_double,1);
    settings.Frames = size(imgs_3d_double,3);
    trackedPar = struct;
    for i=1:length(data_cell_array)
      %convert to um:
      trackedPar(1,i).xy =  impars.PixelSize .* data_cell_array{i}(:,1:2);
      trackedPar(i).Frame = data_cell_array{i}(:,3);
      trackedPar(i).TimeStamp = impars.FrameRate.* data_cell_array{i}(:,3);
    end
    disp(['Localized and tracked ', num2str(length(trackedPar)), ' trajectories']);
    save([output_path, Filenames{iter}, '_Tracked.mat'], 'trackedPar', 'settings');
    toc;
    % Producing data for dashboard
    tracks = struct('data', []);
    for n = 1 : numel(trackedPar)
      tracks(n) = struct('data', [trackedPar(n).TimeStamp, trackedPar(n).xy]);
    end

    rows_to_remove = [];
    for n = 1 : numel(tracks)
      if size(tracks(n).data, 1) < traj_length
        rows_to_remove = [rows_to_remove; n];
      end
    end
    tracks(rows_to_remove) = [];

    trackedData = ["TF_number" "Frame" "x_coord" "y_coord"];
    for n = 1 : length(tracks)
      track_length = size(tracks(n).data, 1);
      trackedData = [trackedData; [repelem(n, track_length)' tracks(n).data(:, 1)./impars.FrameRate tracks(n).data(:, 2:3)]];
    end
    writematrix(trackedData, strcat(output_path_further_processing, "fast_", extractBefore(Filenames(iter),".tif")), 'Delimiter', ',')
    clear imgs_3d_matrix imgs_3d_double data_cell_array trackedPar
    disp('-----------------------------------------------------------------');
  end
end

% Compiling data if ran with multi-core
files_to_remove = [];
if runParallel
  p = gcp;
  delete(p)
  for iter = 1:length(Filenames)
    disp(iter)
    %%%%%%%% SAVE THE TRAJECTORIES TO YOUR STRUCTURED ARRAY FORMAT %%%%%%%%
    tic;
    disp(['saving MATLAB workspace for movie ', num2str(iter), ' of ', num2str(length(Filenames))]);
    data_cell_array = temp_struct_for_save(iter).data_cell_array;
    if length(data_cell_array) < min_traj
      files_to_remove = [files_to_remove; iter];
      continue
    end
    % save the name:
    impars.name = Filenames{iter};
    % save meta-data
    settings.Delay = impars.FrameRate;
    settings.px2micron = impars.PixelSize;
    settings.TrackingOptions = trackpars;
    settings.LocOptions = locpars;
    settings.AcquisitionOptions = impars;
    settings.Filename = impars.name;
    settings.Width = temp_struct_for_save(iter).Width;
    settings.Height = temp_struct_for_save(iter).Height;
    settings.Frames = temp_struct_for_save(iter).Frames;
    trackedPar = struct;
    for i=1:length(data_cell_array)
      %convert to um:
      trackedPar(1,i).xy =  impars.PixelSize .* data_cell_array{i}(:,1:2);
      trackedPar(i).Frame = data_cell_array{i}(:,3);
      trackedPar(i).TimeStamp = impars.FrameRate.* data_cell_array{i}(:,3);
    end
    disp(['Localized and tracked ', num2str(length(trackedPar)), ' trajectories']);
    save([output_path, Filenames{iter}, '_Tracked.mat'], 'trackedPar', 'settings');
    toc;
    % Producing data for dashboard
    tracks = struct('data', []);
    for n = 1 : numel(trackedPar)
      tracks(n) = struct('data', [trackedPar(n).TimeStamp, trackedPar(n).xy]);
    end

    rows_to_remove = [];
    for n = 1 : numel(tracks)
      if size(tracks(n).data, 1) < traj_length
        rows_to_remove = [rows_to_remove; n];
      end
    end
    tracks(rows_to_remove) = [];

    trackedData = ["TF_number" "Frame" "x_coord" "y_coord"];
    for n = 1 : length(tracks)
      track_length = size(tracks(n).data, 1);
      trackedData = [trackedData; [repelem(n, track_length)' tracks(n).data(:, 1)./impars.FrameRate tracks(n).data(:, 2:3)]];
    end
    writematrix(trackedData, strcat(output_path_further_processing, "fast_", Filenames{iter}), 'Delimiter', ',')
    clear imgs_3d_matrix imgs_3d_double data_cell_array trackedPar
    disp('-----------------------------------------------------------------');
  end
end
Filenames(files_to_remove) = [];
tEnd = toc(tWhole)
clear input_path Boule_free tWhole tEnd sig_free T T_off tif_files ...
      runParallel numCores i im_t img_iter Nb_combi Nb_STK nbImages ...
      settings impars iter locpars trackpars Poids_melange_aplha ...
      Poids_melange_diff data nblmages t_red min_traj ...
      tracks rows_to_remove trackedData track_length ...
      files_to_remove

%% Producing slow acquisition data
if acquisition_rate == "slow"
  for k = 1:numel(file_names)
    fnm = fullfile([output_path_further_processing, file_names{k}, '_table.txt']);
    txt = importdata(fnm);
    index = unique(txt(:,4));
    [p,~] = size(index);

    for i=1:p
      logi=find(txt(:,4)==index(i));
      tracklength(i) = (txt(logi(end),3)-txt(logi(1),3)+1);
    end
    tracklength = (ExposureTime / 1000) * tracklength;
    writematrix(tracklength', [fnm(1:end-4) '_dwelltime.txt'], 'delimiter', '\t');
    a = tracklength;
    binsize = 1;
    binwindows = 0:binsize:1000;
    [n,~]=hist(a,binwindows);
    cdf_n = [sum(n), sum(n) - cumsum(n)]/sum(n);
    logind=find(cdf_n>0.01);
    binwindows = 0:binsize:(logind(end)+1)*binsize;
    [n,xout]=hist(a,binwindows);
    histogram(a, binwindows);
    %   hist(a,binwindows);
    cdf_n = [sum(n), sum(n) - cumsum(n)]/sum(n);
    cdf_n=cdf_n(1:end-1);

    % xout is the time, n is the count.
    % Please define the bleach rate (unit in second) from previsous measurement
    myfun = @(x) cdf_n(1:end) - x(1) - x(2)*exp(-xout(1:end)/x(3));%1-component fitting
    myfun2 = @(r) cdf_n - r(1)*exp(-xout/r(2))-(1-r(1))*exp(-xout/r(3));%2-component fitting
    x0=[0.05, 1, 5];
    r0=[0.5,8,1.2];
    options = optimset('Algorithm',{'levenberg-marquardt',0.000001});
    x=lsqnonlin(myfun,x0,[],[],options);
    r=lsqnonlin(myfun2,r0,[],[],options);
    figure;
    hold on
    plot(xout, cdf_n,'o');
    xfit=xout(1):0.9:max(xout);
    yfit= x(1) + x(2)*exp(-xfit/x(3));
    plot(xfit,yfit,'-r');
    title('1-Component Fitting', 'interpreter', 'latex');
    hold off
    f1=figure;
    hold on
    plot(xout, cdf_n,'o');
    xfit=0:0.9:max(xout);
    yfit= r(1)*exp(-xfit/r(2))+(1-r(1))*exp(-xfit/r(3));
    plot(xfit,yfit,'-r')
    a = r(1);
    %   b = r(2);
    c = r(3);
    TrueR = (bleach_rate - 1)/(bleach_rate/x(3) -1);
    TrueR2 = (bleach_rate - 1)/(bleach_rate/max(r(2),r(3)) -1);
    title('2-Components Fitting', 'interpreter', 'latex');
    polyfit_str = ['R1= ' num2str(TrueR2) ' Fraction= ' num2str(a)...
      'R2= ' num2str(c)];
    text(1,1,polyfit_str,'FontSize',18);

    saveas(f1,strcat(fnm, "_fig", ".tif"));

    fprintf('The residence time is %d before correction of one-componen fitting.\n', x(3));
    fprintf('The true residence time is %d of one component fitting after correction.\n', TrueR);
    fprintf('The residence time are %d and %d with fraction %d of two-componen fitting.\n', r(2),r(3),r(1));
    fprintf('The true residence time is %d of two component fitting after correction.\n', TrueR2);

    clear tracklength;

    full_row = {fnm TrueR2 c a};
    df = full_row;
    if k == 1
      df_final = df;
    else
      df_final =[df_final;df];
    end

    dwellTimeData = [TrueR2, c, a];
    save([output_path, file_names{k}, '_dwellTime.mat'], 'dwellTimeData');

  end

%   % Convert cell to a table and use first row as variable names
%   T = cell2table(df_final);
% 
%   % Write the table to a CSV file
%   writetable(T,[pathname 'LongDwell_data.csv']);
end
clear output_path_further_processing

%% Producing diffusion data
tracked_files = cell(numel(Filenames), 1);
for n = 1 : numel(Filenames)
  tracked_files{n} = [Filenames{n} '_Tracked.mat'];
end
subPlotNumbers = ceil(sqrt(length(tracked_files)));

figure;
for iter = 1:length(tracked_files)
  maxJump = 0;
  
  load([output_path, tracked_files{iter}]);
  
  % trackedPar to tracks
  tracks = struct('data', []);
  for n = 1 : numel(trackedPar)
    tracks(n) = struct('data', [trackedPar(n).TimeStamp, trackedPar(n).xy]);
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
    jdTracks = [jdTracks; trackedPar(n).xy, trackedPar(n).Frame, repelem(n, length(trackedPar(n).Frame))', repelem(D(n), length(trackedPar(n).Frame))'];
    if length(trackedPar(n).Frame) > maxJump
      maxJump = length(trackedPar(n).Frame);
    end
  end
  binEdges = 0 : ExposureTime / 1000 : maxJump;
  [tlist, rlist, JDH] = calculateJDH(jdTracks, 1, binEdges, ExposureTime / 1000, clip_factor, analysis_type, 0, 0);
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

%% Plotting diffusion graphs
diffusion_files = cell(numel(Filenames), 1);
for n = 1 : numel(Filenames)
  diffusion_files{n} = [Filenames{n} '_Diffusion.mat'];
end
cell_types = "";

number_of_cells = numel(diffusion_files);
for iter = 1:numel(diffusion_files)
    cell_types{iter} = extractBefore(diffusion_files{iter},"_");
end
cell_types = unique(cell_types);

xbins = linspace(-4,2, bins);
% figure;
y_all = zeros(number_of_cells, size(xbins, 2), size(cell_types, 2));
cell_types_number = zeros(size(cell_types, 2) + 1, 1);
for n = 1:number_of_cells
  load([output_path, diffusion_files{n}]);
  cell_type = find(cell_types == extractBefore(diffusion_files{n},"_"));
  
%   [f,x] = hist(log10(E),xbins);
%   y = f/trapz(x,f);
  f = histcounts(log10(E), xbins);
  y = f/trapz(xbins, [0 f]);
%   plot(x, y, '-+k', 'linewidth', 2); hold on;
  y_all(n, 2:end, cell_type) = y;
  cell_types_number(cell_type + 1) = cell_types_number(cell_type + 1) + 1;
end

figure;
current_count = 0; diffusion_threshold_line = zeros(size(cell_types, 2), 4);
for n = 1:size(cell_types, 2)
  y_all_cell_type = y_all(current_count + 1:current_count + cell_types_number(n + 1), :, n);
  avg_y = mean(y_all_cell_type, 1);
  current_count = current_count + cell_types_number(n + 1);
  p = errorbar(xbins, avg_y, avg_y - min(y_all_cell_type, [], 1), max(y_all_cell_type, [], 1) - avg_y, '-+'); hold on;
%   peaks = findpeaks(avg_y);
%   % if more than 2 peaks, pick 2 biggest peaks
%   if length(peaks) > 2
%     peaks_sorted = sort(peaks);
%     peaks = peaks_sorted(end-1 : end);
%   end
%   xFirstPeak = xbins(avg_y == peaks(1)); xSecondPeak = xbins(avg_y == peaks(2));
%   xMiddlePeak = (xFirstPeak(end) + xSecondPeak(end)) / 2;
%   diffusion_threshold_line(n, :) = [xMiddlePeak p.Color];
end
% for n = 1:size(cell_types, 2)
%   diffusion_threshold = diffusion_threshold_line(n, 1);
%   plot([diffusion_threshold diffusion_threshold], ylim, '--', 'color', diffusion_threshold_line(n, 2:4));
% end
title('Data Variation Error Bar', 'interpreter', 'latex');
xlabel('Log10($D(um^2/s)$)', 'interpreter', 'latex');
ylabel('Density', 'interpreter', 'latex');
legend(cell_types, 'interpreter', 'latex');
xlim([-4, 2]);

figure;
current_count = 0; diffusion_threshold_line = zeros(size(cell_types, 2), 4);
for n = 1:size(cell_types, 2)
  y_all_cell_type = y_all(current_count + 1:current_count + cell_types_number(n + 1), :, n);
  avg_y = mean(y_all_cell_type, 1);
  current_count = current_count + cell_types_number(n + 1);
  p = errorbar(xbins, avg_y, std(y_all_cell_type, 0, 1)/2, std(y_all_cell_type, 0, 1)/2, '-+'); hold on;
%   peaks = findpeaks(avg_y);
%   % if more than 2 peaks, pick 2 biggest peaks
%   if length(peaks) > 2
%     peaks_sorted = sort(peaks);
%     peaks = peaks_sorted(end-1 : end);
%   end
%   xFirstPeak = xbins(avg_y == peaks(1)); xSecondPeak = xbins(avg_y == peaks(2));
%   xMiddlePeak = (xFirstPeak(end) + xSecondPeak(end)) / 2;
%   diffusion_threshold_line(n, :) = [xMiddlePeak p.Color];
end
% for n = 1:size(cell_types, 2)
%   diffusion_threshold = diffusion_threshold_line(n, 1);
%   plot([diffusion_threshold diffusion_threshold], ylim, '--', 'color', diffusion_threshold_line(n, 2:4));
% end
title('Standard Deviation Error Bar', 'interpreter', 'latex');
xlabel('Log10($D(um^2/s)$)', 'interpreter', 'latex');
ylabel('Density', 'interpreter', 'latex');
legend(cell_types, 'interpreter', 'latex');
xlim([-4, 2]);

figure;
current_count = 0; diffusion_threshold_line = zeros(size(cell_types, 2), 4);
for n = 1:size(cell_types, 2)
  y_all_cell_type = y_all(current_count + 1:current_count + cell_types_number(n + 1), :, n);
  avg_y = mean(y_all_cell_type, 1);
  current_count = current_count + cell_types_number(n + 1);
  p = errorbar(xbins, avg_y, std(y_all_cell_type, 0, 1)/(2 * sqrt(cell_types_number(n + 1))), std(y_all_cell_type, 0, 1)/(2 * sqrt(cell_types_number(n + 1))), '-+'); hold on;
%   peaks = findpeaks(avg_y);
%   % if more than 2 peaks, pick 2 biggest peaks
%   if length(peaks) > 2
%     peaks_sorted = sort(peaks);
%     peaks = peaks_sorted(end-1 : end);
%   end
%   xFirstPeak = xbins(avg_y == peaks(1)); xSecondPeak = xbins(avg_y == peaks(2));
%   xMiddlePeak = (xFirstPeak(end) + xSecondPeak(end)) / 2;
%   diffusion_threshold_line(n, :) = [xMiddlePeak p.Color];
end
% for n = 1:size(cell_types, 2)
%   diffusion_threshold = diffusion_threshold_line(n, 1);
%   plot([diffusion_threshold diffusion_threshold], ylim, '--', 'color', diffusion_threshold_line(n, 2:4));
% end
title('Standard Error of Mean Error Bar', 'interpreter', 'latex');
xlabel('Log10($D(um^2/s)$)', 'interpreter', 'latex');
ylabel('Density', 'interpreter', 'latex');
legend(cell_types, 'interpreter', 'latex');
xlim([-4, 2]);

clear avg_y bins current_count diffusion_files diffusion_threshold...
      diffusion_threshold_line E f iter n p peaks peaks_sorted ...
      tol x xbins y y_all y_all_cell_type

%% Plotting msd graphs
msd_files = cell(numel(Filenames), 1);
for n = 1 : numel(Filenames)
  msd_files{n} = [Filenames{n} '_msd.mat'];
end
number_of_cells = numel(msd_files);

max_time = 0; total_traj = 0;
for n = 1 : number_of_cells
  load([output_path, msd_files{n}]);
  if size(msd{1}, 1) > max_time
    max_time = size(msd{1}, 1);
  end
  total_traj = total_traj + size(msd, 1);
end
msd_all = zeros(total_traj, max_time);

cell_types_number = zeros(size(cell_types, 2) + 1, 1);
cell_types_traj_count = zeros(size(cell_types, 2) + 1, 1);
current_line = 1;
for n = 1 : number_of_cells
  load([output_path, msd_files{n}]);
  cell_type = find(cell_types == extractBefore(msd_files{n},"_"));
  for m = 1 : size(msd, 1)
    msd_all(current_line, :) = [msd{m}(:, 2)' nan(1, max_time - length(msd{m}(:, 2)'))];
    current_line = current_line + 1;
  end
  cell_types_number(cell_type + 1) = cell_types_number(cell_type + 1) + 1;
  cell_types_traj_count(cell_type + 1) = cell_types_traj_count(cell_type + 1) + size(msd, 1);
end

figure;
current_line = 1;
time_plot = 0 : ExposureTime / 1000 : (ExposureTime / 1000) * (max_time - 1);
for n = 1:size(cell_types, 2)
  mean_msd_plot = mean(msd_all(current_line : current_line + cell_types_traj_count(n + 1) - 1, :), 1, "omitnan");
  std_msd_plot = std(msd_all(current_line : current_line + cell_types_traj_count(n + 1) - 1, :), 1, "omitnan");
  current_line = current_line + cell_types_traj_count(n + 1);
%   plot(time_plot(1:10), mean_msd_plot(1:10)); hold on;
  errorbar(time_plot(1:10), mean_msd_plot(1:10), std_msd_plot(1:10)./2, std_msd_plot(1:10)./2, '-+'); hold on;
end
title('Average Mean Square Displacement', 'interpreter', 'latex');
xlabel('Time $(s)$', 'interpreter', 'latex');
ylabel('Mean Square Displacement $(\mu m ^ 2)$', 'interpreter', 'latex');
legend(cell_types, 'interpreter', 'latex');

clear cell_type cell_types cell_types_number ExposureTime m max_time ...
      mean_msd_plot msd msd_all msd_files n number_of_cells output_path ...
      temp_msd time_plot total_traj current_line cell_types_traj_count

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