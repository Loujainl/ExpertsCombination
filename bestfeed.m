clear all;

%% fix seed for random number generator (to have repeatable results)
rng(1);
%% "small n, large p" toy example
n = 20;      %number of training data
p = 1000;     %number of coefficients or features
p_star  = 150;
disp(['Number of Relevant Features:',num2str(p_star)]);

%generate unknown, true coefficient values (only first p_star are non-zero)
w_star = [randn(p_star, 1); zeros(p-p_star,1)];

experts_nu = 5;
disp(['Number of Experts:', num2str(experts_nu)]);

% number of questions to ask equal number of relevant features
budget = p_star;

 %create n, p dimensional training data
    x = randn(n,p);
    y = normrnd(x*w_star, 1);

    %create some test data
    x_test = randn(1000,p);
    y_test = normrnd(x_test*w_star, 1);

    %initialize the inputs
    pr  = struct('tau2', 1^2 , 'eta2',0.1^2,'p_u', 0.95, 'rho', p_star/p, ...
        'sigma2_prior',true,'sigma2_a',1,'sigma2_b',1 );
    op = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0,...
        'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
    %% Feedback part: here you specify the feedback from the expert(s)
    % about *relevance of coefficients* given : 
    % ->  0 if the expert thinks  feature  "not relevant"  
    % ->  1 if the expert thinks  feature  "relevant" 

    % create random binary matrix of all experts feedbacks
    % variables, set % of 1s for each expert
    percentage_of_1 = [0.77 0.7 0.65 0.5 0.4];
    all_feedbacks = zeros(experts_nu,budget); 
    % generate feedback accordingly 
    for i=1:experts_nu 
        feedback_per_expert = zeros(1, budget);
        % change to 1 the right amount of feedbacks
        feedback_per_expert(1:(percentage_of_1(i)*budget)) = 1;
        % random permiutations of the 1s and 0s to make sure that the not only 
        % the first features are each time the ones with correct feedback
        feedback_per_expert = feedback_per_expert(randperm(length(feedback_per_expert)));
        all_feedbacks(i,:) = feedback_per_expert;
    end

    all_feedbacks = double(all_feedbacks');


    % calculating expert confidality 
     experts_level = mean(all_feedbacks,1);

    feedback = zeros(2,p)';
    MSE_with_multi_fb = zeros(experts_nu,1); %error per expert


        for j = 1:experts_nu
            feedback = [[all_feedbacks(1:budget,j); zeros(p-budget,1)], [1:p]' ];
            [fa_fb, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], feedback, []);
            MSE_with_multi_fb(j) = mean((x_test*fa_fb.w.Mean- y_test).^2); 

        end



        % Clustering features 
        field_size = round(budget/experts_nu); 
        % numbre of fields = number of experts
        field_acc = zeros(experts_nu);
        for ex=1:experts_nu
            tempo = all_feedbacks(:,ex)';
        %calculating accuracy per field for each expert
            field_acc(:,ex) = mean(reshape(tempo(1:field_size * floor(numel(tempo) / field_size)), [], field_size), 2);
        end
    [maxx, m_indx] = max(field_acc,[],2);
% retrieving the best feedback from max field_accuracy
    best_feed = zeros(budget,1);
    for count = 1: experts_nu
        best_feed(((count-1)*field_size)+1: count*field_size,:) = all_feedbacks(((count-1)*field_size)+1: count*field_size , m_indx(count));
        % best_feed(x-1)*field_size)+1: x*field_size) = [best_feed; temp];
    end

    best_feed = [[best_feed(1:budget,:); zeros(p-budget,1)], [1:p]' ];
    best_feed_level =mean(best_feed(1:budget,1),1);
    [fa_fb, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], best_feed, []);
    MSE_with_best_field_fb = mean((x_test*fa_fb.w.Mean- y_test).^2); 
    disp(['Spike-and-slab field feedback feedback:',num2str(MSE_with_best_field_fb)]);    
   

% apply feedback in ridge regression
% apply feedback on all the features
% distribute between 
% compute diversity of experts
% c_bound