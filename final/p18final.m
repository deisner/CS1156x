% edX ML Final Exam problem 18
% Author: David Eisner (deisner@gmail.com)

function [Ein_zero_avg, Ein_avg] = p18final()
    
    N_reps = 2000;
    N_data = 100;
    
    K = 9;
    gamma = 1.5;
    
    % testing
    % gamma = 2;
    
    N_reject = 0;
    N_runs   = 0;  % might be less than N_reps!
    
    N_Ein_zero = 0;
    Ein_sum = 0;
    
    h = waitbar(0, 'Running (0 %) ...');
    
    for rep=1:N_reps
        Xtrain = gen_data(N_data);
        N_train = size(Xtrain,1);
        
        [Mu, reject] = get_centers(Xtrain(:,1:2), K);
        if reject
            N_reject = N_reject + 1;
            continue;  
        end
        
            
        % Train
        Phi = PHI(Xtrain(:,1:2), Mu, gamma);

        % w = inv(Phi'*Phi)*(Phi'*Xtrain(:,3));
        w = (Phi'*Phi)\(Phi'*Xtrain(:,3));
        
        % Calculat E_in
        y_rbf = zeros(N_train,1);
        for i=1:N_train
           y_rbf(i) = rbf_classify( Xtrain(i,1:2), w, Mu, gamma); 
        end
        Ein = sum(y_rbf ~= Xtrain(:,3))/numel(y_rbf);
        Ein_sum = Ein_sum + Ein;
        
        if sum(y_rbf ~= Xtrain(:,3)) == 0
           N_Ein_zero = N_Ein_zero + 1; 
        end

        N_runs = N_runs + 1;
        if mod(rep, N_reps/100) == 0
            waitbar(rep/N_reps, h, sprintf('Running (%d %%) ...', 100*rep/N_reps)); 
        end
    end
    
    close(h);
    
    Ein_avg = Ein_sum/N_runs;
    % Calculate averages
    Ein_zero_avg = N_Ein_zero/N_runs;
    
    fprintf('**** Lloyd empty cluster ratio: %f\n', N_reject/N_reps);
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
