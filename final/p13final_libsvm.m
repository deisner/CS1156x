% EdX CS1156x Learning from Data Final Exam, Problem 13 - LIBSVM version
% Author: David Eisner (deisner@gmail.com)

function [p_non_sep, N_non_sep] = p13final_libsvm()
%     epsl = 1e-7;
%     opts = optimoptions('quadprog', 'Display', 'off');
    
    N_reps = 1000;
    N_data = 100;
    gamma = 1.5;
    svmopts = sprintf('-s 0 -t 2 -g %f -c 1e6 -q', gamma);
    h = waitbar(0, 'Running...');
    N_non_sep = 0;
    for rep=1:N_reps
        Xtrain = gen_data(N_data);
        model = svmtrain( Xtrain(:,3), Xtrain(:,1:2), svmopts);
        ylib = svmpredict(Xtrain(:,3), Xtrain(:,1:2), model, '-q');

        N_diff = sum( ylib ~= Xtrain(:,3));
        % Ein = N_diff/numel(yqp);
        % fprintf( '*** Ein, ndiff: %f, %d\n', Ein, N_diff);
        if N_diff > 0
            N_non_sep = N_non_sep + 1;
        end
        
        if mod(rep, N_reps/100) == 0
           waitbar(rep/N_reps, h, 'Running ...'); 
        end
    end
    
    close(h);
    p_non_sep = N_non_sep / N_reps;
end


% Generates X = [x1_1 x2_1 y1; ...; x1_N x2_N yN]
% N: number of points to generate
function X = gen_data(N)
    X1 = 2*rand(N,1) - 1;
    X2 = 2*rand(N,1) - 1;
    Y = sign(X2 - X1 + 0.25*sin(pi*X1));
    X = [X1 X2 Y];
    
    % XXX temp: add some noise as a sanity check to see if we can
    %           fail to classify the data, ever.
    % set 5 random points to -1
    % X( randi([1,N], 5,1)) = -1;
end
