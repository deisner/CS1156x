% EdX CS1156x Learning from Data Final Exam, Problem 13 - QP version
% Author: David Eisner (deisner@gmail.com)

function p_non_sep = p13final()
    epsl = 1e-7;
    opts = optimoptions('quadprog', 'Display', 'off');
    
    N_reps = 1000;
    N_data = 100;
    gamma = 1.5;
    
    h = waitbar(0, 'Running...');
    N_non_sep = 0;
    for rep=1:N_reps
        Xtrain = gen_data(N_data);
        N = size(Xtrain,1);
        Q = Qd(Xtrain, gamma);
        A = Ad(Xtrain(:,3));

        alpha = quadprog(Q, -ones(N,1), -A, -zeros(N+2,1), ...
            [], [], [], [], [], opts);

        alpha_sv = alpha(alpha >= epsl);
        Xtrain_sv = Xtrain(alpha >= epsl,:);
        N_sv = size(alpha_sv,1);

        % s is index of support vector to use for calculating b.
        % We pick the largest alpha just to avoid border cases
        [~, s] = max(alpha_sv);

        % calculate b from QP result
        b = Xtrain_sv(s,3) - ...
               (Xtrain_sv(:,3) .* alpha_sv)' * ...
               K( Xtrain_sv(:,1:2), repmat(Xtrain_sv(s,1:2), N_sv, 1), gamma);


        yqp = zeros(N,1);
        for i=1:N
            yqp(i) = classify_pt(Xtrain(i,1:2), Xtrain_sv, alpha_sv, b, gamma); 
        end

        N_diff = sum( yqp ~= Xtrain(:,3));
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

% pt: row vector
function y = classify_pt(pt, Xtrain_sv, alpha_sv, b, gamma)
    
    N_sv = size(alpha_sv,1);

    % calculate sum in one fell swoop
    y = (Xtrain_sv(:,3).*alpha_sv)' * K(Xtrain_sv(:,1:2),repmat(pt,N_sv,1),gamma);
    y = sign( y + b);
end

% X = [x1_1 x2_1 y1; x1_2 x2_2 y2; ...; x1_N x2_N yN]
function Q = Qd(X, gamma)

    N = size(X,1);
    Q = zeros(N);  % pre-allocate matrix
    for i=1:N
        for j=1:N
            Q(i,j) = X(i,3)*X(j,3)*K( X(i,1:2), X(j,1:2), gamma);
        end
    end
end

% % x1, x2 column vectors
% function k = Kold(x1, x2, gamma) 
% 
%     k = exp(-gamma*(x1-x2)'*(x1-x2));
% end


% XA and XB are Mx2 matrices
% Treats each row of XA and XB as a 2-vector
% and returns a column vector of RBF values computed
% for each row.
function KV = K(XA, XB, gamma) 

     KV = exp(-gamma*sum((XA - XB).*(XA-XB),2));
end



function A = Ad(y)
    N = size(y,1);
    A = [y'; -y'; eye(N)];
end

% Generates X = [x1_1 x2_1 y1; ...; x1_N x2_N yN]
% N: number of points to generate
function X = gen_data(N)
    X1 = 2*rand(N,1) - 1;
    X2 = 2*rand(N,1) - 1;
    Y = sign(X2 - X1 + 0.25*sin(pi*X1));
    X = [X1 X2 Y];
end
