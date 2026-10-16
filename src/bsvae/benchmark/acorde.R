#!/usr/bin/env Rscript
## acorde-style isoform co-expression clustering on one expression matrix
## (features x samples), following Arzalluz-Luque et al. (2022):
##   1. correlation matrix (Pearson on bulk samples; the original percentile
##      correlation needs cell-type groups and reduces to this when there is one group),
##   2. average-linkage hclust on 1 - |cor| cut with dynamicTreeCut::cutreeHybrid
##      (deepSplit 3, pamStage FALSE, minClusterSize 20),
##   3. filter_clusters: drop transcripts with >= 3 within-cluster correlations
##      below 0.8, then clusters of size <= 3,
##   4. merge_clusters: complete-linkage on 1 - cor(metatranscripts), cut at 0.2,
##   5. expand_clusters: unclustered transcripts join the cluster whose
##      metatranscript they correlate with most (forced, as in the pipeline).
##
## Usage:
##   Rscript acorde.R --expr expr.csv.gz --outdir out [--deep-split 3] [--min-cluster-size 20]
##          [--min-cor 0.8] [--lowcor-threshold 3] [--size-threshold 3]
##          [--merge-height 0.2] [--expand false] [--signed false] [--seed 42]
##
## Bulk note: the pipeline's 0.8 correlation filter and forced expansion were
## tuned for percentile correlations on single-cell data; on bulk Pearson
## correlations the Python runner defaults to min_cor 0.5 and no expansion
## (unclustered transcripts stay grey).
##
## Writes modules.csv (feature_id, module, module_label; grey -> -1) and params.json.

suppressPackageStartupMessages(library(dynamicTreeCut))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(expr = NULL, outdir = NULL, `deep-split` = 3, `min-cluster-size` = 20,
            `min-cor` = 0.8, `lowcor-threshold` = 3, `size-threshold` = 3,
            `merge-height` = 0.2, expand = "false", signed = "false", seed = 42)
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
flag <- function(x) tolower(as.character(x)) %in% c("true", "1", "yes")
set.seed(int(opt$seed))
dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
log_msg <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S]"), sprintf(...), "\n")

expr <- read.csv(gzfile(opt$expr), row.names = 1, check.names = FALSE)
X <- as.matrix(expr)                      # features x samples
feature_ids <- rownames(X)
keep <- apply(X, 1, stats::sd) > 0
log_msg("acorde-style clustering: %d features x %d samples (%d constant dropped)", nrow(X), ncol(X), sum(!keep))
Xk <- X[keep, , drop = FALSE]
cors <- stats::cor(t(Xk))
signed <- flag(opt$signed)
dist_mat <- if (signed) 1 - cors else 1 - abs(cors)

## 2. dynamic clustering
h <- stats::hclust(stats::as.dist(dist_mat), method = "average")
hdyn <- dynamicTreeCut::cutreeHybrid(h, dist_mat, deepSplit = int(opt$`deep-split`), pamStage = FALSE,
                                     minClusterSize = int(opt$`min-cluster-size`), verbose = 0)
labels <- hdyn$labels                     # 0 = unclustered
names(labels) <- rownames(Xk)
clusters <- split(names(labels), labels)
unclustered <- if ("0" %in% names(clusters)) clusters[["0"]] else character(0)
clusters <- clusters[names(clusters) != "0"]
log_msg("cutreeHybrid: %d clusters, %d unclustered", length(clusters), length(unclustered))

## 3. filter clusters
min_cor <- num(opt$`min-cor`); lowcor <- int(opt$`lowcor-threshold`); size_thr <- int(opt$`size-threshold`)
filtered <- lapply(clusters, function(cl) {
    cc <- cors[cl, cl, drop = FALSE]
    if (!signed) cc <- abs(cc)
    ok <- apply(cc, 2, function(col) sum(col < min_cor) < lowcor)
    cl[ok]
})
dropped <- unlist(mapply(function(f, c) c[!(c %in% f)], filtered, clusters, SIMPLIFY = FALSE))
unclustered <- c(unclustered, dropped)
small <- vapply(filtered, length, 1L) <= size_thr
unclustered <- c(unclustered, unlist(filtered[small]))
filtered <- filtered[!small]
log_msg("filter_clusters: %d clusters kept, %d unclustered", length(filtered), length(unclustered))

## 4. merge clusters by metatranscript correlation
if (length(filtered) >= 2) {
    meta <- sapply(filtered, function(cl) colMeans(Xk[cl, , drop = FALSE]))
    cm <- stats::cor(meta); cm[cm < 0] <- 0
    hm <- stats::hclust(stats::as.dist(1 - cm), method = "complete")
    grp <- stats::cutree(hm, h = num(opt$`merge-height`))
    merged <- lapply(split(seq_along(filtered), grp), function(ix) unlist(filtered[ix]))
} else {
    merged <- filtered
}
log_msg("merge_clusters: %d clusters after merging", length(merged))

## 5. expand: unclustered transcripts join the best-correlated metatranscript
if (flag(opt$expand) && length(unclustered) > 0 && length(merged) >= 1) {
    meta <- sapply(merged, function(cl) colMeans(Xk[cl, , drop = FALSE]))
    uc <- stats::cor(t(Xk[unclustered, , drop = FALSE]), meta)
    best <- apply(uc, 1, which.max)
    for (j in seq_along(merged)) merged[[j]] <- c(merged[[j]], unclustered[best == j])
    unclustered <- character(0)
}

module <- rep(-1L, length(feature_ids)); names(module) <- feature_ids
for (j in seq_along(merged)) module[merged[[j]]] <- j - 1L
write.csv(data.frame(feature_id = feature_ids, module = as.integer(module),
                     module_label = ifelse(module < 0, "grey", paste0("M", module))),
          file.path(opt$outdir, "modules.csv"), row.names = FALSE, quote = FALSE)
params <- sprintf('{"n_modules": %d, "n_grey": %d, "deep_split": %d, "min_cluster_size": %d, "merge_height": %s, "signed": %s}',
                  length(merged), sum(module < 0), int(opt$`deep-split`), int(opt$`min-cluster-size`),
                  num(opt$`merge-height`), tolower(as.character(signed)))
writeLines(params, file.path(opt$outdir, "params.json"))
log_msg("done: %d modules, %d grey", length(merged), sum(module < 0))
