#!/usr/bin/env Rscript
## WGCNA module detection on one expression matrix (features x samples).
##
## Usage:
##   Rscript wgcna.R --expr expr.csv.gz --outdir out [--network-type signed]
##          [--soft-power P] [--min-module-size 10] [--deep-split 2]
##          [--merge-cut-height 0.25] [--blockwise-threshold 20000]
##          [--max-block-size 5000] [--threads 1] [--seed 42] [--fixed-k 0]
##
## --fixed-k K > 0 replaces the dynamic tree cut by a Ward (ward.D2) clustering of the TOM
## dissimilarity cut into K clusters (no merge step, no size floor): WGCNA's network at a forced
## resolution. Cutting the average-linkage tree or PAM on the same dissimilarity is not used:
## both peel off singleton outliers and leave the large clusters whole.
##
## Writes modules.csv (feature_id, module, module_label; grey -> -1),
## eigengenes.csv (samples x modules) and params.json.

suppressPackageStartupMessages(library(WGCNA))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(expr = NULL, outdir = NULL, `network-type` = "signed", `soft-power` = NA,
            `min-module-size` = 10, `deep-split` = 2, `merge-cut-height` = 0.25,
            `blockwise-threshold` = 20000, `max-block-size` = 5000, threads = 1, seed = 42, `fixed-k` = 0)
i <- 1
while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (!(key %in% names(opt))) stop(sprintf("Unknown option: %s", args[i]))
    opt[[key]] <- args[i + 1]
    i <- i + 2
}
if (is.null(opt$expr) || is.null(opt$outdir)) stop("--expr and --outdir are required")
num <- function(x) as.numeric(x)
int <- function(x) as.integer(round(as.numeric(x)))

set.seed(int(opt$seed))
dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
log_msg <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S]"), sprintf(...), "\n")

n_threads <- int(opt$threads)
if (!is.na(n_threads) && n_threads > 1) enableWGCNAThreads(nThreads = n_threads) else disableWGCNAThreads()

network_type <- opt$`network-type`
if (!(network_type %in% c("signed", "unsigned", "signed hybrid"))) stop("network-type must be signed|unsigned|signed hybrid")
tom_type <- if (network_type == "unsigned") "unsigned" else "signed"

log_msg("Loading %s", opt$expr)
con <- if (grepl("\\.gz$", opt$expr)) gzfile(opt$expr) else file(opt$expr)
dat <- read.csv(con, row.names = 1, check.names = FALSE)
datExpr <- t(as.matrix(dat))                      # samples x features
feature_ids <- colnames(datExpr)
n_feat <- ncol(datExpr)
log_msg("Data: %d samples x %d features", nrow(datExpr), n_feat)

gsg <- goodSamplesGenes(datExpr, verbose = 0)
if (!gsg$allOK) {
    log_msg("Removing %d features / %d samples flagged by goodSamplesGenes",
            sum(!gsg$goodGenes), sum(!gsg$goodSamples))
    datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes, drop = FALSE]
}
kept_ids <- colnames(datExpr)

## Soft-threshold selection
soft_power <- suppressWarnings(as.numeric(opt$`soft-power`))
if (is.na(soft_power)) {
    powers <- c(1:10, seq(12, 30, by = 2))
    sft <- pickSoftThreshold(datExpr, powerVector = powers, networkType = network_type, verbose = 0)
    r2 <- -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2]
    cand <- powers[!is.na(r2) & r2 >= 0.85]
    soft_power <- if (length(cand) > 0) min(cand) else if (network_type == "unsigned") 6 else 12
    log_msg("Auto-selected soft power %d", soft_power)
} else {
    log_msg("Using soft power %d", soft_power)
}

min_size <- int(opt$`min-module-size`)
deep_split <- int(opt$`deep-split`)
cut_height <- num(opt$`merge-cut-height`)
fixed_k <- int(opt$`fixed-k`)
if (is.na(fixed_k)) fixed_k <- 0L
if (fixed_k > 0 && n_feat > int(opt$`blockwise-threshold`)) stop("--fixed-k needs the single-block path (n_feat <= blockwise-threshold)")

if (n_feat > int(opt$`blockwise-threshold`)) {
    log_msg("Blockwise WGCNA (maxBlockSize=%d)", int(opt$`max-block-size`))
    net <- blockwiseModules(
        datExpr, power = soft_power, networkType = network_type, TOMType = tom_type,
        minModuleSize = min_size, deepSplit = deep_split, mergeCutHeight = cut_height,
        maxBlockSize = int(opt$`max-block-size`), numericLabels = TRUE,
        pamRespectsDendro = FALSE, randomSeed = int(opt$seed), verbose = 0
    )
    labels <- net$colors
    MEs <- net$MEs
} else {
    log_msg("Adjacency + TOM")
    adj <- adjacency(datExpr, power = soft_power, type = network_type)
    TOM <- TOMsimilarity(adj, TOMType = tom_type, verbose = 0)
    dissTOM <- 1 - TOM
    tree <- hclust(as.dist(dissTOM), method = "average")
    if (fixed_k > 0) {
        log_msg("Ward clustering of the TOM dissimilarity cut into %d clusters", fixed_k)
        labels <- as.integer(cutree(hclust(as.dist(dissTOM), method = "ward.D2"), k = min(fixed_k, n_feat)))
        MEs <- moduleEigengenes(datExpr, colors = labels)$eigengenes
    } else {
    dyn <- cutreeDynamic(dendro = tree, distM = dissTOM, deepSplit = deep_split,
                         pamRespectsDendro = FALSE, minClusterSize = min_size, verbose = 0)
    if (length(unique(dyn[dyn != 0])) >= 2) {
        merged <- mergeCloseModules(datExpr, dyn, cutHeight = cut_height, verbose = 0)
        labels <- merged$colors
        MEs <- merged$newMEs
    } else {
        labels <- dyn
        MEs <- moduleEigengenes(datExpr, colors = labels)$eigengenes
    }
    }
}

## Relabel to 0..M-1 with grey (0) -> -1
uniq <- sort(unique(labels[labels != 0]))
relabel <- setNames(seq_along(uniq) - 1, uniq)
module <- ifelse(labels == 0, -1L, as.integer(relabel[as.character(labels)]))
module_label <- ifelse(module < 0, "grey", paste0("M", module))

## Features dropped by goodSamplesGenes are grey.
all_module <- rep(-1L, length(feature_ids)); names(all_module) <- feature_ids
all_module[kept_ids] <- module
all_label <- ifelse(all_module < 0, "grey", paste0("M", all_module))

write.csv(data.frame(feature_id = feature_ids, module = as.integer(all_module),
                     module_label = all_label, stringsAsFactors = FALSE),
          file.path(opt$outdir, "modules.csv"), row.names = FALSE, quote = FALSE)

me_names <- sub("^ME", "", colnames(MEs))
keep <- me_names != "0"
MEs_out <- MEs[, keep, drop = FALSE]
colnames(MEs_out) <- paste0("ME", as.integer(relabel[me_names[keep]]))
MEs_out <- cbind(data.frame(sample_id = rownames(datExpr), stringsAsFactors = FALSE), MEs_out)
write.csv(MEs_out, file.path(opt$outdir, "eigengenes.csv"), row.names = FALSE, quote = FALSE)

params <- sprintf('{"soft_power": %d, "network_type": "%s", "min_module_size": %d, "deep_split": %d, "merge_cut_height": %s, "fixed_k": %d, "n_modules": %d, "n_grey": %d, "blockwise": %s}',
                  as.integer(soft_power), network_type, min_size, deep_split, format(cut_height), fixed_k,
                  length(uniq), sum(all_module < 0), tolower(as.character(n_feat > int(opt$`blockwise-threshold`))))
writeLines(params, file.path(opt$outdir, "params.json"))
log_msg("Done: %d modules, %d grey", length(uniq), sum(all_module < 0))
