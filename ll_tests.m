clear all;

%% fix seed for random number generator (to have repeatable results)

rng(1);
%% "small n, large p" toy example
n = 10;      %number of training data
p = 1000;     %number of coefficients or features
p_star  = 50;
disp(['Number of Relevant Features:',num2str(p_star)]);

%generate unknown, true coefficient values (only first p_star are non-zero)
w_star = [randn(p_star, 1); zeros(p-p_star,1)];

experts_nu = 5;
disp(['Number of Experts:', num2str(experts_nu)]);

% number of questions to ask equal number of relevant features
budget = p_star;
run_times = 100;
%start big loop
MSE = zeros(run_times, experts_nu);
for iter= 1:run_times

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
        %Assume we have feedback about *relevance of coefficients* given
        %by an expert (this is a matrix of dimension p*2 (where p is the number of 
        %features, on the 2nd column we have the feature number 1...p, on the first 
        %column we have the feedback : 
        % ->  0 if the expert thinks  feature  "not relevant"  
        % ->  1 if the expert thinks  feature  "relevant" 



        % create random binary matrix of all experts feedbacks
        % variables, set % of 1s for each expert
        percentage_of_1 = [0.8 0.7 0.6 0.5 0.4];
        all_feedbacks = zeros(experts_nu,budget); 
        % generate feedback accordingly 
        for i=1:experts_nu 
            feedback_per_expert = zeros(1, budget);
            % change to 1 the right amount of feedbacks
            feedback_per_expert(1:(percentage_of_1(i)*budget)) = 1;
            % random permiutations of the 1s and 0s to male sure that the not only 
            % the first features are each time the ones with correct feedback
            feedback_per_expert = feedback_per_expert(randperm(length(feedback_per_expert)));
            all_feedbacks(i,:) = feedback_per_expert;
        end

        all_feedbacks = all_feedbacks';
        %all_feedbacks = randsrc(budget,experts_nu,[0 1; .2 .8]);

        % create increasing number of 1s in the feedback (accuracy)
        %thresVec = linspace(0.45,0.75,experts_nu);  %# thresholds increasing accuracy between 0.45 & 0.8 
        %all_feedbacks = bsxfun(@lt,rand(budget,experts_nu),thresVec); %# vectors are per column
        all_feedbacks = double(all_feedbacks);



        % calculating expert confidality 
        experts_level = mean(all_feedbacks,1);
       
        %%exp_lev = zeros(experts_nu);
        feedback = zeros(2,p)';
        MSE_with_multi_fb = zeros(experts_nu,1); %error per expert

            for j = 1:experts_nu
                %exp_lev(j) = sum(all_feedbacks(:,j))/budget;
                feedback = [[all_feedbacks(1:budget,j); zeros(p-budget,1)], [1:p]' ];
                [fa_fb, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], feedback, []);
                MSE_with_multi_fb(j) = mean((x_test*fa_fb.w.Mean- y_test).^2); 
                %disp(['Spike-and-slab with user feedback ',num2str(j),' = ',num2str(MSE_with_multi_fb(j))]);

            end
MSE(iter,:)=MSE_with_multi_fb;
end

    MSE_avg = mean(MSE,1);
    disp(['Spike-and-slab with user feedback  = ',num2str(MSE_avg)]);

    plot(experts_level,MSE_avg);
    xlabel('Expert Confidality');
    ylabel('Error');
    hold on;


    %majority vote
        vote = mean(all_feedbacks,2);
        majority_feedback = zeros(length(vote),1);
        for i= 1:length(vote)
            if vote(i) > 0.5 
                  majority_feedback(i) = 1;
            else 
                majority_feedback(i) = 0;
            end
        end
    majority_feedback = [[majority_feedback; zeros(p-budget,1)], [1:p]' ];
    [fa_fb, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], majority_feedback, []);
    MSE_with_majority_fb = mean((x_test*fa_fb.w.Mean- y_test).^2); 
    disp(['Spike-and-slab majority feedback:',num2str(MSE_with_majority_fb)]);
    majority_confidality =mean(experts_level) ;
plot(majority_confidality,MSE_with_majority_fb,'r*','MarkerSize',10);
    % Clustering features 
    field_size = round(budget,experts_nu);
    field_acc = zeros(experts_nu);
    for j = 1:experts_nu
        temp = all_feedbacks(:,j);
        for k =1:budget
            field_acc(j) = arrayfun(@(i) mean(all_feedbacks(i:i+field_size-1)) ,1:field_size:length(all_feedbacks)-field_size+1)';
        end
    end
    %disp(['Field accuracies:',num2str(field_acc)])


%results without feedback (only spike and slab model)
[fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], [], []);
MSE_without_fb = mean((x_test*fa.w.Mean- y_test).^2);
%ridge regression solution
w_ridge = inv(eye(p) + (x'*x)) * (x'*y);
MSE_ridge = mean((x_test*w_ridge- y_test).^2);

disp('Mean Squared Error on test data:')

disp(['Spike-and-slab without user feedback:',num2str(MSE_without_fb)])
disp(['Ridge regression:',num2str(MSE_ridge)])



