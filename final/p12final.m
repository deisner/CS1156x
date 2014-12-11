% EdX CS1156x Learning from Data Final Exam, Problem 12
% Author: David Eisner (deisner@gmail.com)

function [alpha, model] = p12final()

X = [1 0 -1; 0 1 -1; 0 -1 -1; -1 0 1; 0 2 1; 0 -2 1; -2 0 1];
N = size(X,1);
Q = Qd(X);
A = Ad(X(:,3));
alpha = quadprog(Q, -ones(N,1), -A, -zeros(N+2,1));

% repeat with LIBSVM, for comparison
svmopts = '-s 0 -t 1 -d 2 -g 1 -r 1 -c 1e6';
model = svmtrain( X(:,3), X(:,1:2), svmopts);
end

% function v = K(x1, x2) 
%     v = (1+x1'*x2)^2;
% end

% X = [x1_1 x2_1 y1; x1_2 x2_2 y2; ...; x1_N x2_N yN]
function Q = Qd(X)

    N = size(X,1);
    Q = zeros(N);  % pre-allocate matrix
    for i=1:N
        for j=1:N
            Q(i,j) = X(i,3)*X(j,3)*(1+X(i,1:2)*X(j,1:2)')^2;     
        end
    end
end

function A = Ad(y)
    N = size(y,1);
    A = [y'; -y'; eye(N)];
end
