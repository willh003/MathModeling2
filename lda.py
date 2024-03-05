import numpy as np
import os
import matplotlib.pyplot as plt
from helpFunctions import loadMulti, getPix, showHistograms


def load_data(df):
    multi_paths = [f for f in os.listdir('data') if f.endswith('.mat')]
    
    all_multis = []
    all_anns = []
    all_rgbs = []
    for multi in multi_paths:

        day = multi.split('_')[1].split('.')[0]
        annotation = f'annotation_{day}.png'
        color = f'color_{day}.png'
        data = loadMulti(multi, annotation, df)

        multi_im, ann_im = data
        rgb = plt.imread(os.path.join(df,color))
        all_multis.append(multi_im)
        all_anns.append(ann_im) 
        all_rgbs.append(rgb)

    return np.stack(all_multis, axis=0), np.stack(all_anns, axis=0), np.stack(all_rgbs, axis=0)

def model_metrics(model, test_data, labels):
    """
    Returns a list containing the metrics of the model evaluated on test_data
    Metrics supported are 
        - Accuracy
        - Precision (2 class)
        - Recall (2 class)
    Accuracy is defined as (TP + TN) / (TP + TN + FP + FN)
    """
    _, _, n_classes = labels.shape

    trues = 0
    falses = 0

    for i in range(n_classes):
        label = labels[:, :, i]
        pix_vals, _, _ = getPix(test_data, label)

        preds = model.forward(pix_vals)
        n_correct = (preds == i).sum()
        n_incorrect = (preds != i).sum()
        trues += n_correct
        falses += n_incorrect

        if i == 0:
            fn = falses
        if i == 1:
            precision = trues / (trues + falses)
            recall = trues / (trues + fn)
    accuracy = trues / (trues + falses)
    f1 = 2 * (precision * recall) / (precision + recall)
    metric_vals = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return metric_vals

class MultiBandGaussianDiscriminant():
    """
    A gaussian discriminant model for multiple spectral bands
    """
    def __init__(self, num_classes, feature_dim, prior = None, pooled_cov = False):
        """
        Prior must have shape (num_classes,)
        """
        self.num_classes = num_classes
        self.pooled_cov = pooled_cov
        self.prior = prior if prior is not None else np.ones(num_classes) / num_classes
        self.mus = np.zeros((num_classes, feature_dim))
        self.cov = np.eye(feature_dim)

    def fit(self, multi, anns):
        mus = []
        covs = []

        _,_, c = anns.shape
        assert c == self.num_classes, f"Error: expecting {self.num_classes} classes in the annotation image"
        
        total_observations = 0
        pooled_cov = np.zeros((multi.shape[-1], multi.shape[-1]))
        for i in range(c):
            ann = anns[:, :, i]
            pix_vals, r,c = getPix(multi, ann)
            mu = np.mean(pix_vals, axis=0)
            cov = np.cov(pix_vals.T)
            mus.append(mu)
            covs.append(cov)

            pooled_cov += (len(pix_vals) - 1) * cov
            total_observations += len(pix_vals) - 1
        
        pooled_cov /= total_observations

        self.mus = np.stack(mus)
        if self.pooled_cov:
            self.cov = pooled_cov
    


    def multivariate_gaussian_eval(self, mu, cov, x):
        """
        Returns the multivariate gaussian probability of x given mu, cov
        """
        d = len(mu)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm = 1/(np.sqrt((2*np.pi)**d * det))
        diff = (x - mu)

        exponent= -.5*((diff @ inv)*diff).sum(axis=-1)
        return norm * np.exp(exponent)
    
    def multivariate_gaussian_discriminant(self,mu,cov, x, p_i=1):
        """
        Calculate the discriminant function for a multivariate gaussian (difference in the log of the pdf for two classes)
        Requires: prior != 0
        """
        inv = np.linalg.inv(cov)
        log_prior = np.log(p_i) if p_i!=0 else 0
        d = ((x @ inv)*mu).sum(axis=-1) - .5*((mu @ inv)*mu).sum(axis=-1) + log_prior
        return d

    def forward(self, spectrum, prior=None):
        """
        Returns the class predictions given gaussian discriminant parameters mus, covs, and prior
        Spectrum has shape (h, w, l) or (h*w, l)
        """
        ps = []
        shape = len(spectrum.shape)
        if shape == 3:
            h, w,l = spectrum.shape
            spectrum = spectrum.reshape(-1, spectrum.shape[-1])
        
        for mu, p_i in zip(self.mus, self.prior):
            p = self.multivariate_gaussian_discriminant(mu, self.cov, spectrum, p_i)
            ps.append(p)
        
        classes = np.argmax(np.stack(ps, axis=0), axis=0)

        if shape ==3:
            classes = classes.reshape(h, w)
        return classes

class SingleBandGaussianDiscriminant:
    """
    A Gaussian Discriminant Model for a single spectral band
    """
    def __init__(self, num_classes, update_sigma = False):
        self.mus = np.zeros(num_classes)
        self.sigmas = np.ones(num_classes)
        self.update_sigma = update_sigma # False to assume constant variance
        self.num_classes = num_classes

    def fit(self, multi, anns, band_idx=0):
        """
        Returns the means and standard deviations for the pixels in each class given by anns
        """

        mus = []
        sigmas = []

        _,_, c = anns.shape
        assert c == self.num_classes, f"Error: expecting {self.num_classes} classes in the annotation image"
        
        for i in range(c):
            ann = anns[:, :, i]
            pix, _, _ = getPix(multi, ann)
            mu = np.mean(pix[:, band_idx])
            sigma = np.std(pix[:, band_idx])
            mus.append(mu)
            sigmas.append(sigma)
        
        self.mus = np.stack(mus)
        if self.update_sigma:
            self.sigmas = np.stack(sigmas)


    def forward(self, spectral_layer):
        """
        Returns the class predictions given gaussian discriminant parameters mus, sigmas
        """
        ps = []
        h, w = spectral_layer.shape
        
        for mu, sigma in zip(self.mus, self.sigmas):
            p = self.gaussian_eval(mu, sigma, spectral_layer.flatten())
            ps.append(p)
        
        classes = np.argmax(np.stack(ps, axis=0), axis=0)

        classes = classes.reshape(h, w)
        return classes

    def gaussian_eval(self, mu, sigma, x):
        return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

    def get_threshold(self):
        """
        Returns the decision threshold
        Requires: self.num_classes == 2 and self.update_sigma == False
        """
        assert self.num_classes == 2 and not self.update_sigma, "Closed form decision boundary only implemented for 2 classes with constant variance"
        return (self.mus[1]**2 - self.mus[0]**2) / (2 * (self.mus[1] - self.mus[0]))


def dice_coef(y_true, y_pred, smooth=1):
    """
    Compute the dice coefficient
    
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (np.sum(y_true_f**2) + np.sum(y_pred_f**2) + smooth)


def spectral_histogram(multi, masks, classes=None):
    """
    Plots a histogram of the values in each spectral layer in multi, masked according to mask
    """
    fig, axes= plt.subplots(multi.shape[-1])

    for i, ax in enumerate(axes):
        for j, mask in enumerate(masks):
            to_plot = (multi[:, :, i][np.nonzero(mask)]).flatten()
            ax.hist(to_plot, bins=100, range=(0, 80), label=classes[j] if classes else None)
    axes[0].legend()
    plt.show()

def plot_masked_spectral(spectral_layer, pred_mask, gt_mask):
    """
    Plots a spectral layer with a mask overlay, with predictions next to ground truth
    """
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(spectral_layer, cmap='gray')
    pos1 = ax1.imshow(pred_mask, cmap='jet', alpha=0.5)
    ax1.set_title('Predictions')
    fig.colorbar(pos1)    

    ax2.imshow(spectral_layer, cmap='gray')
    pos2 = ax2.imshow(gt_mask, cmap='jet', alpha=0.5)
    ax2.set_title('Ground Truth')
    fig.colorbar(pos2)    

    plt.show()


def single_band_multi_label(multis, anns):
    band_idx = 0 
    train_img_idx = 4
    test_img_idx = 4

    train_annotations = anns[train_img_idx,:,:,:] 
    non_salami = np.prod(1-train_annotations, axis=2)
    all_train_annotations = np.concatenate([np.expand_dims(non_salami, axis=2), train_annotations], axis=2)

    single_band = SingleBandGaussianDiscriminant(num_classes=4, update_sigma=False)
    single_band.fit(multis[train_img_idx], all_train_annotations, band_idx=band_idx)

    test_annotations = anns[test_img_idx,:,:,:] 
    non_salami = np.prod(1-test_annotations, axis=2)
    all_test_annotations = np.concatenate([np.expand_dims(non_salami, axis=2), test_annotations], axis=2)

    preds = single_band.forward(multis[test_img_idx, :, :, band_idx])

    label_encode = np.argmax(all_test_annotations, axis=2)

    plot_masked_spectral(multis[test_img_idx, :, :, band_idx], preds, label_encode)


def single_band_exp(multis, anns, band_idx = 0, train_img_idx = 0, test_img_idx = 0):

    train_annotations = anns[train_img_idx,:,:,1:]
    test_annotations = anns[test_img_idx,:,:,1:] 
 
    model = SingleBandGaussianDiscriminant(num_classes=2, update_sigma=False)
    model.fit(multis[train_img_idx], train_annotations, band_idx=band_idx)

    preds = model.forward(multis[test_img_idx, :, :, band_idx])

    label_encode = np.argmax(test_annotations, axis=2)
    thresh = model.get_threshold()
    print(f'Decision threshold: {thresh}')

    metrics = model_metrics(model, multis[test_img_idx], test_annotations)

    print(f'Accuracy on test annotations: {metrics["accuracy"]:.3f}')
    print(f'Precision score on test annotations: {metrics["precision"]:.3f}')
    print(f'Recall on test annotations: {metrics["recall"]:.3f}')
    print(f'F1 score on test annotations: {metrics["f1"]:.3f}')

    plot_masked_spectral(multis[test_img_idx, :, :, band_idx], preds, label_encode)
    return thresh


def inspect_distributions(multis, anns, img_idx):
    """
    Show the amounts of fat and meat in this image, and plot a spectral histogram of meat and fat
    """
    fat_mask = anns[img_idx, :, :, 1]
    meat_mask = anns[img_idx, :, :, 2]
    
    print(f'Fat count: {np.sum(fat_mask)}')
    print(f'Meat count: {np.sum(meat_mask)}')
    print(f'Fat ratio: {np.sum(fat_mask) / (np.sum(fat_mask) + np.sum(meat_mask)):.3f} ')

        
    spectral_histogram(multis[img_idx],masks=[fat_mask, meat_mask], classes=['Fat', 'Meat'])

def multi_band_exp(multis, anns, rgbs, train_img_idx, test_img_idx):
    model = MultiBandGaussianDiscriminant(num_classes=2, 
                                        feature_dim=multis[train_img_idx].shape[-1], 
                                        pooled_cov=True,
                                        prior=np.array([.3,.7]))
    
    train_annotations = anns[train_img_idx,:,:,1:]
    test_annotations = anns[test_img_idx,:,:,1:] 
    model.fit(multis[train_img_idx], train_annotations)
    metrics = model_metrics(model, multis[test_img_idx], test_annotations)

    print(f'Accuracy on test annotations: {metrics["accuracy"]:.3f}')
    print(f'Precision score on test annotations: {metrics["precision"]:.3f}')
    print(f'Recall on test annotations: {metrics["recall"]:.3f}')
    print(f'F1 score on test annotations: {metrics["f1"]:.3f}')

    preds = model.forward(multis[test_img_idx])
    label_encode = np.argmax(anns[test_img_idx], axis=2)
    plot_masked_spectral(rgbs[test_img_idx], preds, label_encode)


def both_models_all_images_exp(data, anns, rgbs, single_band_idx=0):
    """
    Train both models on each day and test on all other days
    Plot the performance of the different models depending on the training day
    """
    multi_accs = []
    single_accs = []
    for train_idx in range(len(data)):
        single_model = SingleBandGaussianDiscriminant(num_classes=2, update_sigma=False)        
        multi_model = MultiBandGaussianDiscriminant(num_classes=2, 
                                            feature_dim=data[train_idx].shape[-1], 
                                            pooled_cov=True,
                                            prior=np.array([.3,.7]))
        

        train_annotations = anns[train_idx,:,:,1:]
        multi_model.fit(data[train_idx], train_annotations)
        single_model.fit(data[train_idx], train_annotations, band_idx=single_band_idx)

        multi_acc = []
        single_acc = []

        for test_idx in range(len(data)):
            if test_idx == train_idx:
                continue
            test_annotations = anns[test_idx,:,:,1:]

            multi_metrics = model_metrics(multi_model, data[test_idx], test_annotations)  
            multi_acc.append(multi_metrics['accuracy'])
            
            single_metrics = model_metrics(single_model, data[test_idx], test_annotations)  
            single_acc.append(single_metrics['accuracy'])

        single_accs.append(single_acc)
        multi_accs.append(multi_acc)

    single_accs = np.array(single_accs)
    multi_accs = np.array(multi_accs)

    print('Single and Multi Band Accuracies')
    print('table[i, j] is accuracy of model trained on day i and tested on day j')
    print(np.stack((single_accs, multi_accs)))

    plt.plot(range(len(multi_accs)), multi_accs.mean(axis=1), label='Multi Band')
    plt.plot(range(len(single_accs)), single_accs.mean(axis=1), label='Single Band')
    plt.xlabel('Training Image Index')
    plt.ylabel('Average Training Accuracy on Non-Training Images')
    plt.title('Performance of Single and Multi Band Models on Different Training Days')
    plt.legend()
    plt.show()


    
def main():
    df='data'
    multis, anns, rgbs = load_data(df)
    multi_band_exp(multis, anns, rgbs, 0, 0)

    #single_band_exp(multis, anns, band_idx=12, train_img_idx=0, test_img_idx=1)
    #inspect_distributions(multis, anns, 0)


    # TODO: find the best single_band_idx, and do this experiment with that one
    #both_models_all_images_exp(multis,anns, rgbs, single_band_idx=0)








if __name__=='__main__':
    main()