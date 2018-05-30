%clear all;

%% fix seed for random number generator (to have repeatable results)
rng(1);
%% "small n, large p" toy example
n = 10;      %number of training data
p = 1000;     %number of coefficients or features
p_star  = 120;
disp(['Number of training data:', num2str(n)]);
disp(['Number of features:', num2str(p)]);

disp(['Number of Relevant Features:',num2str(p_star)]);

%generate unknown, true coefficient values (only first p_star are non-zero)
w_star = [randn(p_star, 1); zeros(p-p_star,1)];

experts_nu = 5;
disp(['Number of Experts:', num2str(experts_nu)]);

% number of questions to ask equal number of relevant features
budget = p_star;
run_times = 100;
disp(['Number of runtimes: ',num2str(run_times)]);

%start big loop
MSE = zeros(run_times, experts_nu);
MSE_maj = zeros(run_times, 1);
MSE_Best = zeros(run_times,1);
MSE_no_fb = zeros(run_times,1);
MSE_ridge_nofb =zeros(run_times,1) ;

for iter= 1:run_times
    
        %create n, p dimensional training data
        x = randn(n,p);
        y = normrnd(x*w_star, 1);
        
        % Normalize  training data
        x_mean = mean(x);
        x_std = std(x);
        x = (x-x_mean)./x_std;
        
        y_mean = mean(y);
        y_std = std(y);
        y = (y-y_mean)./y_std;

        %create some test data
        x_test = randn(1000,p);
        y_test = normrnd(x_test*w_star, 1);
        
        % Normalise test data
        test_mean = mean(x_test);
        test_std = std(x_test);
        x_test = (x_test-test_mean)./test_std;
        
        ytest_mean = mean(y_test);
        ytest_std = std(y_test);
       y_test = (y_test-ytest_mean)./ytest_std;

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
        percentage_of_1 = [0.70 0.6 0.55 0.50 0.4];
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
   % disp(['Spike-and-slab field feedback feedback:',num2str(MSE_with_best_field_fb)]);    
   
    %results without feedback (only spike and slab model)
    [fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, [], [], []);
    MSE_without_fb = mean((x_test*fa.w.Mean- y_test).^2);
    %ridge regression solution
    w_ridge = inv(eye(p) + (x'*x)) * (x'*y);
    MSE_ridge = mean((x_test*w_ridge- y_test).^2);
        
        MSE_maj(iter) = MSE_with_majority_fb ;
        MSE(iter,:) = MSE_with_multi_fb;
        MSE_Best(iter) = MSE_with_best_field_fb; 
        MSE_no_fb(iter) = MSE_without_fb;
        MSE_ridge_nofb(iter) = MSE_ridge; 
        
end
    disp(['Experts Levels:',num2str(experts_level)]);
    MSE_avg = mean(MSE,1);
    disp(['Spike-and-slab with user feedback  = ',num2str(MSE_avg)]);
    MSE_maj_avg = mean(MSE_maj);
    majority_confidality =mean(majority_feedback(1:budget,1),1) ;
    disp([' Majority Vote accuracy level  = ',num2str(majority_confidality)]);
    disp(['Spike-and-slab with Majority Vote feedback  = ',num2str(MSE_maj_avg)]);
    MSE_Best_avg = mean(MSE_Best);
    disp(['Best Field accuracy level  = ',num2str(best_feed_level)]);

    disp(['Spike-and-slab with Best Field feedback  = ',num2str(MSE_Best_avg)]);
    
    
    %plot(experts_level,MSE_avg,'b--o');
    plot(1:run_times,MSE(:,1),'--bo','LineWidth',2,'MarkerSize',10);
    xlabel('Iteration');
    ylabel('Error');
    hold on;
 

   % plot(majority_confidality,MSE_maj_avg,'r*','MarkerSize',10);
    plot(1:run_times,MSE_maj,'r-*','LineWidth',2,'MarkerSize',10);
    
    %plot(best_feed_level,MSE_Best_avg,'gx','MarkerSize',12);
    plot(1:run_times,MSE_Best,'k-.d','LineWidth',2,'MarkerSize',10);


    disp('Mean Squared Error on test data:')
    MSE_no_fb_avg = mean(MSE_no_fb);
    disp(['Spike-and-slab without user feedback:',num2str(MSE_no_fb_avg)])
    plot(0,MSE_no_fb_avg,'ks','MarkerSize',16);
    
    MSE_ridge_nofb_avg = mean(MSE_ridge_nofb);
    disp(['Ridge regression:',num2str(MSE_ridge_nofb_avg)])
    plot(0,MSE_ridge_nofb_avg,'rs','MarkerSize',16);
    %legend(series, 'Location', 'NorthWest');



% % Results: 
%Number of training data:10
%Number of features:1000
%Number of Relevant Features:120
%Number of Experts:5
%Number of runtimes: 100
%Experts Levels:0.7         0.6        0.55         0.5         0.4
%Spike-and-slab with user feedback  = 0.9543     0.96592     0.98264     0.99413      1.0246
% Majority Vote accuracy level  = 0.59167
%Spike-and-slab with Majority Vote feedback  = 0.97254
%Best Field accuracy level  = 0.63333
%Spike-and-slab with Best Field feedback  = 0.95462
%Mean Squared Error on test data:
%Spike-and-slab without user feedback:0.99066
%Ridge regression:0.99066
%

