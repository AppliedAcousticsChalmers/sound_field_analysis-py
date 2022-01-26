% requires AKtools toolbox (run AKtoolsStart.m)
% $ svn checkout https://svn.ak.tu-berlin.de/svn/AKtools --username aktools --password ak
%
% requires Spherical-Harmonic-Transform scripts
% $ git clone https://github.com/polarch/Spherical-Harmonic-Transform.git
%
% requires soundfieldsynthesis "Common" scripts
% $ git clone https://github.com/JensAhrens/soundfieldsynthesis.git
%
% requires subplot_er()
% (install "Border-less tight subplot (auto-refresh)" in Add-On Explorer)
%
% optionally requires distFig()
% (install "Distribute figures" by Anders Simonsen in Add-On Explorer)
%
% requires Python environment to compare respective implementations
% (Every different Matlab versions is only compatible with specific Python
% versions, so make sure you install a matching version according to
% https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf 
% A restriction to `python >=3.7,<3.9` in "environment_test.yml" should 
% allow for the compatability with `MATLAB >=R2019a`.)
% $ conda env create --file environment_test.yml --force
%
% Activate the Python environment in Matlab (e.g. MacOS or Windows)
% $ pyversion('~/miniconda3/envs/sfa_compare_SH_conventions/bin/python')
% $ pyversion('%HOMEPATH%\Miniconda3\envs\sfa_compare_SH_conventions\python.exe')
%
close all; clear; clc;

addpath(genpath('tools'));

%%
global STR_SEP IS_PLOT_2D DO_PLOT_EXPORT
STR_SEP = '==================================\n';
IS_PLOT_2D = false;
% IS_PLOT_2D = true;
DO_PLOT_EXPORT = true;
% DO_PLOT_EXPORT = false;

global PLOT_DIR PLOT_RES
PLOT_DIR = 'plots';
PLOT_RES = 2; % degrees

N_max = 4;

%%
tic; % start measuring execution time

% get the coefficients of the SH basis functions
R_N_base = ones((N_max+1)^2,1);

fprintf([STR_SEP, 'Compare complex SHs according to\n', ...
    'Rafaely, B. (2015). Fundamentals of Spherical Array Processing, ', ...
    '(J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg, ', ...
    '2nd ed., 196 pages. doi:10.1007/978-3-319-99561-8\n', STR_SEP]);
plot_coeffs('SH basis', R_N_base, 'sfa-py_complex', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'SFS_complex_wo_cs', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'AKT_complex', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'SHT_complex', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'spaudiopy_complex', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'Scipy_complex', IS_PLOT_2D);
print_halt;

fprintf([STR_SEP, 'Compare complex SHs according to\n', ...
    'Gumerov, N. A., and Duraiswami, R. (2005). Fast Multipole Methods ', ...
    'for the Helmholtz Equation in Three Dimensions, Elsevier Science, ', ...
    'Amsterdam, NL, 520 pages. doi:10.1016/B978-0-08-044371-3.X5000-5\n', STR_SEP]);
plot_coeffs('SH basis', R_N_base, 'sfa-py_complex_GumDur', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'SFS_complex', IS_PLOT_2D);
print_halt;

fprintf([STR_SEP, 'Compare real SHs according to\n', ...
    'Williams, E. G. (1999). Fourier Acoustics: Sound Radiation and ', ...
    'Nearfield Acoustical Holography, (E. G. Williams, Ed.) Academic Press, ', ...
    'London, UK, 1st ed., 1–306 pages. doi:10.1016/B978-012753960-7/50001-2\n', STR_SEP]);
plot_coeffs('SH basis', R_N_base, 'sfa-py_real', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'SFS_real_wikipedia', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'SHT_real', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'spaudiopy_real', IS_PLOT_2D);
print_halt;

fprintf([STR_SEP, 'Compare real SHs according to\n', ...
    'Zotter, F. (2009). Analysis and Synthesis of Sound-Radiation with ', ...
    'Spherical Arrays University of Music and Performing Arts Graz, ', ...
    'Austria, 192 pages.\n', STR_SEP]);
plot_coeffs('SH basis', R_N_base, 'sfa-py_real_Zotter', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'SFS_real', IS_PLOT_2D);
plot_coeffs('SH basis', R_N_base, 'AKT_real', IS_PLOT_2D);
print_halt;

fprintf(' ... finished in %.0fh %.0fm %.0fs.\n', ...
    toc/3600, mod(toc,3600)/60, mod(toc,60));


%% helper functions
function plot_coeffs(name, coeff_N, impl, is_2d)
    global DO_PLOT_EXPORT PLOT_DIR PLOT_RES

    N_max = sqrt(length(coeff_N))-1;
    name = sprintf('%s N=%d (%s)', name, N_max, impl);
    if is_2d
        name = [name, ' 2D'];
    else
        name = [name, ' 3D'];
    end
    if contains(impl, 'complex', 'IgnoreCase', true)
        basis = strsplit(impl, '_');
        basis = strjoin(basis(2:end), '_');
        desc = '[B=real_pos, R=real_neg, C=imag_pos, M=imag_neg]';
    elseif contains(impl, 'real', 'IgnoreCase', true)
        basis = strsplit(impl, '_');
        basis = strjoin(basis(2:end), '_');
        desc = '[B=pos, R=neg]';
    else
        error('Unknown implementation "%s".', impl);
    end

    % get evaluation grid
    if is_2d
        azis_rad = deg2rad(0 : PLOT_RES : 360).';
        cols_rad = ones(size(azis_rad)) * pi / 2;
        dirs_rad = [azis_rad, cols_rad];
    else
        dirs_rad = grid2dirs(PLOT_RES, PLOT_RES); % in flattened shape
        azis_rad = dirs_rad(:, 1);
        cols_rad = dirs_rad(:, 2);
        azis_grid_rad = deg2rad(0 : PLOT_RES : 360);
        cols_grid_rad = deg2rad(0 : PLOT_RES : 180).';
    end

    fprintf('Generating plot "%s" ... ', name);
    fig = figure('Position', [50, 50, 1200, 1000], ...
        'NumberTitle', 'Off', 'Name', name);
    subplot('Position', [0, .6, .3, .3]);

    if contains(impl, 'SHT', 'IgnoreCase', true)
        F = inverseSHT(coeff_N, dirs_rad, basis);
    elseif contains(impl, 'AKT', 'IgnoreCase', true)
        F = AKisht(coeff_N, false, rad2deg(dirs_rad), 'complex', true, true, basis).';
    elseif contains(impl, 'SFS', 'IgnoreCase', true)
        F = sphharm_all(N_max, cols_rad, azis_rad, basis) * coeff_N;
    elseif contains(impl, 'sfa-py', 'IgnoreCase', true)
        F = sfa_sph_harm_all(N_max, azis_rad, cols_rad, basis) * coeff_N;
    elseif contains(impl, 'spaudiopy', 'IgnoreCase', true)
        F = spaudiopy_sph_harm_all(N_max, azis_rad, cols_rad, basis) * coeff_N;
    elseif contains(impl, 'SciPy', 'IgnoreCase', true)
        if strcmpi(basis, 'real')
            error('Real SH basis functions are not implemented in SciPy.');
        end
        F = scipy_sph_harm_all(N_max, azis_rad, cols_rad) * coeff_N;
    else
        error('Unknown implementation "%s".', impl);
    end

    if is_2d
        plot_2d(azis_rad, F);
    else
        % rearrange into grid shape
        F_grid = Fdirs2grid(F, PLOT_RES, PLOT_RES, 1);
        plot_3d(azis_grid_rad, cols_grid_rad, F_grid);
        grid on;
    end
    title({name, desc}, 'Interpreter', 'None');
    drawnow;

    % plot individual SH orders and modes
    for n = 0 : N_max
        for m = -n : n
            vec_id = ((n+1)^2)+m-n;

            ax_id = n*(2*N_max+1)+N_max+1+m;
            ax(vec_id) = subplot_er(N_max+1, 2*N_max+1, ax_id);

            cur_coeff = zeros(size(coeff_N));
            cur_coeff(vec_id) = coeff_N(vec_id);

            if contains(impl, 'SHT', 'IgnoreCase', true)
                F = inverseSHT(cur_coeff, dirs_rad, basis);
            elseif contains(impl, 'AKT', 'IgnoreCase', true)
                F = AKisht(cur_coeff, false, rad2deg(dirs_rad), 'complex', true, true, basis).';
            elseif contains(impl, 'SFS', 'IgnoreCase', true)
                F = sphharm(n, m, cols_rad, azis_rad, basis) * coeff_N(vec_id);
            elseif contains(impl, 'sfa-py', 'IgnoreCase', true)
                F = sfa_sph_harm(m, n, azis_rad, cols_rad, basis) * coeff_N(vec_id);
            elseif contains(impl, 'spaudiopy', 'IgnoreCase', true)
                F = spaudiopy_sph_harm(m, n, azis_rad, cols_rad, basis) * coeff_N(vec_id);
            elseif contains(impl, 'Scipy', 'IgnoreCase', true)
                F = scipy_sph_harm(m, n, azis_rad, cols_rad) * coeff_N(vec_id);
            end

            if is_2d
                plot_2d(azis_rad, F);
            else
                % rearrange into grid shape
                F_grid = Fdirs2grid(F, PLOT_RES, PLOT_RES, 1);
                plot_3d(azis_grid_rad, cols_grid_rad, F_grid);
                axis tight;
                if sum(cur_coeff) == 0; axis([-eps,eps,-eps,eps,-eps,eps]); end
            end

            axis off;
            ax(vec_id) = gca;
        end
    end
    drawnow;

    if is_2d
        r_max = 0;
        for a = 1:length(ax)
            r_max = max([r_max; arrayfun(@(c) max(abs(c.RData)), ax(a).Children)]);
        end
        rlim(ax, [0, r_max]);
    else
        r_max = arrayfun(@(d) max(abs([d.XAxis.Limits, d.YAxis.Limits, d.ZAxis.Limits])), ax);
        r_max = max(r_max);
        axis(ax, [-r_max, r_max, -r_max, r_max, -r_max, r_max]);
    end
    drawnow;

    if DO_PLOT_EXPORT
        fprintf('exporting ... ');

        % create output directory
        [~, ~] = mkdir(fullfile(PLOT_DIR));

        warning('off', 'MATLAB:print:CustomResizeFcnInPrint');
        % export to PNG
        print(fig, fullfile(PLOT_DIR, [name, '.png']), '-dpng', '-r200');
    end

    fprintf('done.\n');
end

function F = numpy2complex(F_nd)
    % individually cast real and imaginary parts
    F = (double(py.array.array('d', py.numpy.nditer(F_nd.real))) ...
        + 1i*double(py.array.array('d', py.numpy.nditer(F_nd.imag)))).';
    % arrange to similar shape
    F = reshape(F, [], F_nd.shape{1}).';
end

function F = sfa_sph_harm(m, n, azis_rad, cols_rad, basis)
    F_nd = py.sound_field_analysis.sph.sph_harm(m, n, azis_rad, cols_rad, basis);
    F = numpy2complex(F_nd);
%     fprintf('sfa_sph_harm       %d,%+.0f     %+f  %+f  %+f  (%f,%f,%f)(%f,%f,%f)\n', n, m, F, azis_rad, cols_rad);
end

function F = sfa_sph_harm_all(N, azis_rad, cols_rad, basis)
    F_nd = py.sound_field_analysis.sph.sph_harm_all(uint8(N), azis_rad, cols_rad, basis);
    F = numpy2complex(F_nd);
end

function F = spaudiopy_sph_harm(m, n, azis_rad, cols_rad, basis)
    % this is very inefficient
    F_all = spaudiopy_sph_harm_all(n, azis_rad, cols_rad, basis);
    F = F_all(:, n^2+n+m+1);
%     fprintf('spaudiopy_sph_harm %d,%+.0f     %+f  %+f  %+f  (%f,%f,%f)(%f,%f,%f)\n', n, m, F, azis_rad, cols_rad);
end

function F = spaudiopy_sph_harm_all(N, azis_rad, cols_rad, basis)
    F_nd = py.spaudiopy.sph.sh_matrix(uint8(N), azis_rad, cols_rad, basis);
    F = numpy2complex(F_nd);
end

function F = scipy_sph_harm(m, n, azis_rad, cols_rad)
    F_nd = py.scipy.special.sph_harm(m, n, azis_rad, cols_rad);
    F = numpy2complex(F_nd);
end

function F = scipy_sph_harm_all(N, azis_rad, cols_rad)
    F = zeros(length(azis_rad), (N+1)^2);
    for n = 0 : N
        for m = -n : n
            F(:, n^2+n+m+1) = scipy_sph_harm(m, n, azis_rad, cols_rad);
        end
    end
end

function plot_2d(azis_rad, F)
    if isreal(F)
        is_complex = false;
    else
        is_complex = true;
        F_imag = imag(F);
        F = real(F);
    end

    polarplot(azis_rad(F>=0), F(F>=0), 'Color', 'b', 'LineWidth', 2);
    hold on;
    polarplot(azis_rad(F<0), -F(F<0), 'Color', 'r', 'LineWidth', 2);

    if is_complex
        polarplot(azis_rad(F_imag>=0), F_imag(F_imag>=0), 'Color', 'c', 'LineWidth', 2);
        polarplot(azis_rad(F_imag<0), -F_imag(F_imag<0), 'Color', 'm', 'LineWidth', 2);
    end

    set(gca, 'ThetaZeroLocation', 'Top');
end

function plot_3d(azis_rad, cols_rad, F)
    is_complex = ~isreal(F);

    % construct real positive and negative parts if real function
    D_x = cos(azis_rad) .* sin(cols_rad) .* abs(F);
    D_y = sin(azis_rad) .* sin(cols_rad) .* abs(F);
    D_z = cos(cols_rad) .* abs(F);
    if is_complex
        Dm_x = D_x;
        Dm_y = D_y;
        Dm_z = D_z;
    else
        Dp_x = D_x.*(F>=0);
        Dp_y = D_y.*(F>=0);
        Dp_z = D_z.*(F>=0);
        Dn_x = D_x.*(F<0);
        Dn_y = D_y.*(F<0);
        Dn_z = D_z.*(F<0);
    end

    % plot function
    hold on;
    if is_complex
        surf(Dm_x, Dm_y, Dm_z, angle(F), 'EdgeColor', 'none');
    else
        Hp = surf(Dp_x, Dp_y, Dp_z);
        Hn = surf(Dn_x, Dn_y, Dn_z);
        set(Hp, 'EdgeColor', 'none', 'FaceColor', 'b', 'FaceAlpha', 0.5);
        set(Hn, 'EdgeColor', 'none', 'FaceColor', 'r', 'FaceAlpha', 0.5);
    %     set(Hp, 'EdgeColor', 'none', 'FaceLighting', 'gouraud', 'FaceColor', 'b');
    %     set(Hn, 'EdgeColor', 'none', 'FaceLighting', 'gouraud', 'FaceColor', 'r');
    %     light('Position', [0 0 1], 'Style', 'infinite');
    %     light('Position', [-1 -1 -1], 'Style', 'infinite');
    %     light('Position', [0 0 -1], 'Style', 'infinite');
    end

    xlabel('x');
    ylabel('y');
    zlabel('z');
    axis equal;
    view(3);
end

function print_halt()
    global STR_SEP

    if exist('distFig', 'file')
        distFig;
    end

    disp('Press key to continue ...')
    pause;

    fprintf([STR_SEP, '\n']);
    close all;
end
