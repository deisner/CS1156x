% edX ML Final Exam problem 17
% Author: David Eisner (deisner@gmail.com)

function [Ntbl] = p17final()
    
    N_reps = 1000;
    N_data = 100;
    N_data_out = 1000;
    
    K = 9;
    gammas = [1.5 2];
    
    % cols: gamma, E_in_sum, E_out_sum
    E_sums = zeros(2,3);
    E_sums(:,1) = gammas';
    
    N_reject = 0;
    N_runs   = 0;  % might be less than N_reps!
    
    % counts of various scenarios, comparing across Kvals(1) to Kvals(2)
    
    N_in_down_out_up    = 0;
    N_in_up_out_down    = 0; 
    N_in_up_out_up      = 0; 
    N_in_down_out_down  = 0;
    N_no_change         = 0;
    
    h = waitbar(0, 'Running (0 %) ...');
    
    for rep=1:N_reps
        Xtrain = gen_data(N_data);
        Xtest  = gen_data(N_data_out);
        
        N_train = size(Xtrain,1);
        N_test  = size(Xtest,1);
        
        % Centers will be same for each gamma       
        [Mu, reject] = get_centers(Xtrain(:,1:2), K);
        if reject
            N_reject = N_reject + 1;
            continue;  
        end
        
        % cols: E_in, E_out; one row for each gamma
        E_vals = zeros(2,2);
        
        gi = 1;
        for gamma = gammas
            
            % Train
            Phi = PHI(Xtrain(:,1:2), Mu, gamma);

            % w = inv(Phi'*Phi)*(Phi'*Xtrain(:,3));
            w = (Phi'*Phi)\(Phi'*Xtrain(:,3));
        
            % Calculat E_in
            y_rbf = zeros(N_train,1);
            for i=1:N_train
               y_rbf(i) = rbf_classify( Xtrain(i,1:2), w, Mu, gamma); 
            end
            E_vals(gi,1) = sum(y_rbf ~= Xtrain(:,3))/numel(y_rbf);


            % Calculate E_out 
            y_rbf = zeros(N_test,1);
            for i=1:N_test
               y_rbf(i) = rbf_classify( Xtest(i,1:2), w, Mu, gamma); 
            end
            E_vals(gi,2) = sum(y_rbf ~= Xtest(:,3))/numel(y_rbf);
            
            gi = gi + 1;
        end % for gamma = gammas
        
        deltaEin =  E_vals(2,1) - E_vals(1,1);
        deltaEout = E_vals(2,2) - E_vals(1,2);
        if deltaEin < 0 && deltaEout > 0
            N_in_down_out_up = N_in_down_out_up + 1;
        elseif deltaEin > 0 && deltaEout < 0
            N_in_up_out_down = N_in_up_out_down + 1;
        elseif deltaEin < 0 && deltaEout < 0
            N_in_down_out_down = N_in_down_out_down + 1;
        elseif deltaEin > 0 && deltaEout > 0
            N_in_up_out_up = N_in_up_out_up + 1;
        elseif deltaEin == 0 && deltaEout == 0
            N_no_change = N_no_change + 1;
        end
        
        E_sums(:,2:3) = E_sums(:,2:3) + E_vals;
        
        N_runs = N_runs + 1;
        if mod(rep, N_reps/100) == 0
            waitbar(rep/N_reps, h, sprintf('Running (%d %%) ...', 100*rep/N_reps)); 
        end
    end
    
    close(h);
    
    % Calculate averages
    E_avgs = E_sums;
    E_avgs(:,2:3) = E_avgs(:,2:3)/N_runs;
    
    fprintf('*** N_runs = %d\n', N_runs);
    fprintf('*** K, E_in_avg, E_out_avg:\n');
    display(E_avgs);
    fprintf('**** Lloyd empty cluster ratio: %f\n', N_reject/N_reps);
    
    Ntbl = [ N_in_down_out_up, N_in_up_out_down, N_in_up_out_up, ...
             N_in_down_out_down, N_no_change ];
end

%   Classify point +1/-1 using regular RBF with K centers
%    pt: 1x2 row vector, the point to classify
%     w: (K+1)x1 column vector of weights, including initial bias w(0)
%    Mu: Kx2 matrix of K centers for RBFs
% gamma: RBF parameter
function y = rbf_classify( pt, w, Mu, gamma)
    K = size(Mu,1);
%     S = 0;
%     for i=1:K
%         S = S + w(i+1)*exp(-gamma*(pt - Mu(i,:))*(pt - Mu(i,:))');
%     end
%     S = S + w(1);
%     y = sign(S);
    X = [1; Krnl( repmat(pt,K,1), Mu, gamma)];
    y = sign( w'*X);
end

% Implements Lloyd's algorithm to calculate K centers from 
% X, an Nx2 matrix of 2-d features: [x1_1 x2_1; ... x1_N x2_N]
%
% Returns Kx2 matrix
function [Mu, reject] = get_centers(X, K)
    N = size(X,1);
    reject = 1;
    
    % Start with K random centers.
    Mu = 2*rand(K,2)-1;
    
    % Clusters: S(i) = k means X(i,:) is in cluster k
    S = zeros(N,1);
    S_new = zeros(N,1);
    
    MAX_LOOP = 500;
    for j=1:MAX_LOOP

        % Update clusters
        for i=1:N
           % tall matrix of distances from X(i,:) to each Mu
           dist = sum( (Mu - repmat( X(i,:), K, 1)).^2, 2); % sum along rows
           [~, min_k] = min(dist);
           S_new(i) = min_k;
        end
                
        % check for empty clusters -- if found, start from scratch
        if numel( unique(S_new)) < K
%            fprintf( '*** empty cluster, restarting\n');
%            fprintf( '*** cluster: ');
%           display(unique(S_new)');
           break;
        end
        
        % Are clusters the same?
        if S_new == S
            reject = 0;
            break;
        end
        S = S_new;
               
        % Update centers
        for k=1:K
            Mu(k,:) = mean( X(S==k,:),1);  % That ,1 is crucial! If cluster
                                           % has only one point, mean sums
                                           % along the single row, giving a
                                           % single number that then gets
                                           % promoted to a vector for the
                                           % assignment. There's a lesson
                                           % here ...
        end
    end
    
    if j == MAX_LOOP
       fprintf( '**** ERROR: INFINITE LOOP in get_centers\n');
       
       display(S);
       display(Mu);
       
       pause(1000);
    end
end


% Calculate matrix PHI for psuedoinverse of RBF with K centers
%
%     X: Nx2 matrix of input coordinates
%    Mu: Kx2 matrix of centers for RBF
% gamma: gamma for RBF
%
% Returns: Nx(K+1) matrix, including 1's in column 1 for bias coordinate
function A = PHI(X, Mu, gamma)
    N = size(X,1);
    K = size(Mu,1);
    
    A = zeros(N, K+1);    % pre-allocate matrix
    
    A(:,1) = 1;
    % calculate A, column-by-column
    for k=1:K
        A(:,k+1) = Krnl(X, repmat(Mu(k,:), N, 1), gamma);
    end
end


% XA and XB are Mx2 matrices
% Treats each row of XA and XB as a 2-vector
% and returns a column vector of RBF values computed
% for each row.
function KV = Krnl(XA, XB, gamma) 

     KV = exp(-gamma*sum((XA-XB).^2,2));
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
