% EdX CS1156x Learning from Data Final Exam, Problem 9
% Author: David Eisner (deisner@gmail.com)

function E_table = p9final()

% data columns: digit, symmetry, intensity
D_train = importdata('features.train');
D_test  = importdata('features.test');

% Din = [din(:,1) din(:,2)];
N_train = size(D_train,1);
N_test  = size(D_test,1);

lambda = 1;
verses = 0:9;

% summarize results in a table. Each row:
% digit, Ein_notrans, Eout_notrans, Ein_trans, Eout_trans
E_table = zeros(numel(verses),5);

i = 1;
for digit = verses
    
    y_train = ones(N_train,1);
    y_train(D_train(:,1) ~= digit) = -1;
   
    y_test = ones(N_test, 1);
    y_test( D_test(:,1) ~= digit) = -1;
    
    Znotrans_train = [ones(N_train,1) D_train(:,2:3)];
    Ztrans_train = phi( D_train(:,2:3));
    
    Znotrans_test = [ones(N_test,1) D_test(:,2:3)];
    Ztrans_test = phi( D_test(:,2:3));
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % First, without transform
    %
    w = linear_reg_decay_w(Znotrans_train, y_train, lambda);
    
    % E_in
    y_pred = sign(Znotrans_train*w);
    E_in_notrans = sum(y_pred ~= y_train)/numel(y_train);
    
    % E_out
    y_pred = sign(Znotrans_test*w);
    E_out_notrans = sum(y_pred ~= y_test)/numel(y_test);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % Now with transform
    %
    w = linear_reg_decay_w(Ztrans_train, y_train, lambda);
    
    % E_in
    y_pred = sign(Ztrans_train*w);
    E_in_trans = sum(y_pred ~= y_train)/numel(y_train);
    
    % E_out
    y_pred = sign(Ztrans_test*w);
    E_out_trans = sum(y_pred ~= y_test)/numel(y_test);
    
    E_table(i,:) = [digit E_in_notrans E_out_notrans E_in_trans E_out_trans];
    i=i+1;
end

disp(E_table);
end

function w_reg = linear_reg_decay_w(Z, y, lambda)
N = size(Z,2);
w_reg = ((Z'*Z + lambda*eye(N))\Z')*y;
end

% D = [x1_1, x2_1; x1_2 x2_2; ... ]
% returns each row of D tranformed to 
% (1, x1, x2, x1*x2, x1^2, x2^2)
function Z = phi(D)
    N = size(D,1);
    X0 = ones(N,1);
    X1 = D(:,1);
    X2 = D(:,2);
    Z = [X0, X1, X2,  X1.*X2, X1.^2, X2.^2];
end


