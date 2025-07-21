using Pkg
#Pkg.activate("~/Downloads/DL-Tutorials")
#Pkg.add(["Lux", "MLDatasets", "Optimisers", "Plots", "MLUtils", "Zygote"])
using Lux, MLDatasets, Optimisers, Random, Statistics, Plots, MLUtils, Zygote

# Load and preprocess data
train_x, train_y = FashionMNIST(:train)[:]
test_x, test_y = FashionMNIST(:test)[:]
train_x = reshape(train_x, 784, :)  # Flatten images (28*28 = 784)
test_x = reshape(test_x, 784, :)
train_x = Float32.(train_x)  # Convert to Float32 for Lux
test_x = Float32.(test_x)
train_y = convert.(Int, train_y)
test_y = convert.(Int, test_y)
train_data = (train_x, train_y)

function make_network(hidden_size)
    model = Chain(
        Dense(784, hidden_size, relu),  # Input to hidden layer
        Dense(hidden_size, 10)  # Hidden to output
    )
    return model
end

# Function to check accuracy
function check_accuracy(model, parameters, state, x_data, y_data)
    predictions, _ = model(x_data, parameters, state)
    predicted_classes = [findmax(predictions[:, i])[2] for i in 1:size(predictions, 2)]
    correct = sum(predicted_classes .== (y_data .+ 1))  # Labels are 0-9
    accuracy = correct / length(y_data)
    return accuracy
end

# Improved training function with fixes
function train_network(model, parameters, state, train_data, test_x, test_y, epochs, batch_size, optimizer_state, learning_rate_schedule=nothing)
    train_x, train_y = train_data
    n_samples = size(train_x, 2)
    
    for epoch = 1:epochs
        # Adjust learning rate if schedule is provided
        if !isnothing(learning_rate_schedule)
            lr = learning_rate_schedule(epoch)
            Optimisers.adjust!(optimizer_state, lr)
        end
        
        # Shuffle indices instead of data
        indices = shuffle(1:n_samples)
        total_loss = 0.0
        n_batches = 0
        
        for i in 1:batch_size:n_samples
            batch_end = min(i + batch_size - 1, n_samples)
            batch_indices = indices[i:batch_end]
            
            x_batch = train_x[:, batch_indices]
            y_batch = train_y[batch_indices]
            y_onehot = onehotbatch(y_batch, 0:9)
            
            # Define loss function with proper gradient computation
            f = p -> begin
                y_pred, s = model(x_batch, p, state)
                # Cross-entropy loss: -sum(y_true * log(softmax(y_pred)))
                y_pred_softmax = softmax(y_pred)
                loss = -sum(y_onehot .* log.(y_pred_softmax .+ 1f-8)) / size(y_onehot, 2)
                (loss, s)
            end
            
            result = Zygote.withgradient(f, parameters)
            loss, new_state = result.val
            grads = result.grad[1]
            
            total_loss += loss
            n_batches += 1
            
            # Update parameters
            optimizer_state, parameters = Optimisers.update(optimizer_state, parameters, grads)
            state = new_state  # Update state for next batch
        end
        
        avg_loss = total_loss / n_batches
        println("Epoch $epoch, Average Loss: $avg_loss")
    end
    return check_accuracy(model, parameters, state, test_x, test_y)
end

# Helper functions for one-hot encoding
function onehot(label, classes)
    vec = falses(length(classes))
    idx = findfirst(==(label), classes)
    if idx !== nothing
        vec[idx] = true
    end
    return vec
end

function onehotbatch(labels, classes)
    return hcat([onehot(label, classes) for label in labels]...)
end

# Softmax function
function softmax(x)
    exp_x = exp.(x .- maximum(x, dims=1))  # Subtract max for numerical stability
    return exp_x ./ sum(exp_x, dims=1)
end

# Experiment 1: Try different hidden layer sizes
hidden_sizes = [10, 20, 40, 50, 100, 300]
accuracies = []
for size in hidden_sizes
    rng = Random.MersenneTwister(1234)
    model = make_network(size)
    parameters, state = Lux.setup(rng, model)
    optimizer = Adam(0.001)
    optimizer_state = Optimisers.setup(optimizer, parameters)
    acc = train_network(model, parameters, state, train_data, test_x, test_y, 10, 32, optimizer_state)
    push!(accuracies, acc)
    println("Hidden size $size, Accuracy: $acc")
end

# Plot for Experiment 1
plot1 = plot(hidden_sizes, accuracies, label="Accuracy", marker=:circle)
xlabel!("Hidden Layer Size")
ylabel!("Test Accuracy")
title!("Accuracy vs Hidden Layer Size")
savefig(plot1, "hidden_size_plot.png")

# Experiment 2: Try different random initializations
hidden_size = 30
num_runs = 10
init_accuracies = []
for run = 1:num_runs
    rng = Random.MersenneTwister(run)
    model = make_network(hidden_size)
    parameters, state = Lux.setup(rng, model)
    optimizer = Adam(0.001)
    optimizer_state = Optimisers.setup(optimizer, parameters)
    acc = train_network(model, parameters, state, train_data, test_x, test_y, 10, 32, optimizer_state)
    push!(init_accuracies, acc)
    println("Run $run, Accuracy: $acc")
end

# Calculate mean and standard deviation
mean_accuracy = sum(init_accuracies) / num_runs
std_accuracy = std(init_accuracies)
println("Mean accuracy: $mean_accuracy, Std dev: $std_accuracy")

# Plot for Experiment 2
plot2 = scatter(1:num_runs, init_accuracies, label="Accuracy")
xlabel!("Run Number")
ylabel!("Test Accuracy")
title!("Accuracy for Different Initializations\nMean: $mean_accuracy, Std: $std_accuracy")
savefig(plot2, "init_plot.png")

# Experiment 3: Train with decaying learning rate
function learning_rate_schedule(epoch)
    lr = 0.001
    if epoch > 5
        lr = lr * 0.1
    end
    if epoch > 10
        lr = lr * 0.1
    end
    return lr
end

rng = Random.MersenneTwister(1234)
model = make_network(30)
parameters, state = Lux.setup(rng, model)
optimizer = Adam(0.001)
optimizer_state = Optimisers.setup(optimizer, parameters)
exp3_accuracy = train_network(model, parameters, state, train_data, test_x, test_y, 25, 32, optimizer_state, learning_rate_schedule)
println("Experiment 3 Accuracy: $exp3_accuracy")

# Experiment 4: Try different batch sizes and learning rates
batch_sizes = [16, 32, 64]
learning_rates = [0.0005, 0.001, 0.005]
results = []
for bs in batch_sizes
    for lr in learning_rates
        rng = Random.MersenneTwister(1234)
        model = make_network(30)
        parameters, state = Lux.setup(rng, model)
        optimizer = Adam(lr)
        optimizer_state = Optimisers.setup(optimizer, parameters)
        acc = train_network(model, parameters, state, train_data, test_x, test_y, 10, bs, optimizer_state)
        push!(results, (bs, lr, acc))
        println("Batch size $bs, LR $lr, Accuracy: $acc")
    end
end

# Find best parameters
best_acc = 0.0
best_batch = 0
best_lr = 0.0
for (bs, lr, acc) in results
    if acc > best_acc
        best_acc = acc
        best_batch = bs
        best_lr = lr
    end
end
println("Best batch size: $best_batch, Best LR: $best_lr, Accuracy: $best_acc")

# Plot for Experiment 4
acc_matrix = zeros(length(batch_sizes), length(learning_rates))
for (i, bs) in enumerate(batch_sizes), (j, lr) in enumerate(learning_rates)
    for (b, l, a) in results
        if b == bs && l == lr
            acc_matrix[i, j] = a
        end
    end
end
plot4 = heatmap(learning_rates, batch_sizes, acc_matrix')  # Transpose for correct axes
xlabel!("Learning Rate")
ylabel!("Batch Size")
title!("Accuracy for Batch Size and LR")
savefig(plot4, "grid_search_plot.png")

# Experiment 5: Train with best parameters and learning rate schedule
learning_rate_sched = let initial_lr = best_lr
    epoch -> begin
        lr = initial_lr
        if epoch > 5
            lr = lr * 0.1
        end
        if epoch > 10
            lr = lr * 0.1
        end
        return lr
    end
end

rng = Random.MersenneTwister(1234)
model = make_network(30)
parameters, state = Lux.setup(rng, model)
optimizer = Adam(best_lr)
optimizer_state = Optimisers.setup(optimizer, parameters)
exp5_accuracy = train_network(model, parameters, state, train_data, test_x, test_y, 25, best_batch, optimizer_state, learning_rate_sched)
println("Experiment 5 Accuracy with Best Parameters: $exp5_accuracy")
println("Did it improve over Experiment 3? ", exp5_accuracy > exp3_accuracy ? "Yes" : "No")
println("Accuracy difference: ", exp5_accuracy - exp3_accuracy)