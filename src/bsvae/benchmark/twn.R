#!/usr/bin/env Rscript
## Transcriptome-wide network (TWN; Saha et al. 2017, battle-lab/twn_tsn)
## re-implemented with glasso instead of the MATLAB QUIC solver.
##
## Nodes are gene total expressions (TE) and isoform ratios (IR); a sparse
## inverse covariance is estimated with an L1 penalty matrix that has one
## value for TE-TE pairs (lambda_tt), TE-IR pairs (lambda_ti), IR-IR pairs
## (lambda_ii), the diagonal (lambda_d) and pairs within the same gene
## (lambda_s: a gene's TE with its own isoforms, and its isoforms with each
## other), exactly as twn.m builds it. Edges are the non-zero off-diagonal
## precision entries; within-gene edges are removed as in filter_edges.R.
##
## Usage:
##   Rscript twn.R --te te.csv.gz --ir ir.csv.gz --tx2gene tx2gene.tsv --outdir out
##          [--lambda-tt 0.5] [--lambda-ti 0.4] [--lambda-ii 0.4] [--lambda-d 0]
##          [--lambda-s 0.05] [--max-iter 100] [--standardize true]
## te: samples x genes, ir: samples x isoforms (same sample order).
## Writes edges.tsv (node1, node2, edge_type (1 TE-TE, 2 TE-IR, 3 IR-IR), weight) and params.json.

suppressPackageStartupMessages(library(glasso))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(te = NULL, ir = NULL, tx2gene = NULL, outdir = NULL, `lambda-tt` = 0.5, `lambda-ti` = 0.4,
            `lambda-ii` = 0.4, `lambda-d` = 0, `lambda-s` = 0.05, `max-iter` = 100, standardize = "true")
i <- 1
while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (!(key %in% names(opt))) stop(sprintf("Unknown option: %s", args[i]))
    opt[[key]] <- args[i + 1]
    i <- i + 2
}
if (is.null(opt$te) || is.null(opt$ir) || is.null(opt$tx2gene) || is.null(opt$outdir)) stop("--te, --ir, --tx2gene and --outdir are required")
num <- function(x) as.numeric(x)
flag <- function(x) tolower(as.character(x)) %in% c("true", "1", "yes")
dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
log_msg <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S]"), sprintf(...), "\n")

te <- as.matrix(read.csv(gzfile(opt$te), row.names = 1, check.names = FALSE))
ir <- as.matrix(read.csv(gzfile(opt$ir), row.names = 1, check.names = FALSE))
if (nrow(te) != nrow(ir) || any(rownames(te) != rownames(ir))) stop("TE and IR must have the same samples in the same order")
t2g <- read.delim(opt$tx2gene, stringsAsFactors = FALSE)
iso2gene <- setNames(as.character(t2g$gene_id), as.character(t2g$transcript_id))
genes <- colnames(te); isoforms <- colnames(ir)
ne <- length(genes); ni <- length(isoforms); n <- ne + ni
M <- cbind(te, ir)
if (flag(opt$standardize)) M <- scale(M)
M[is.na(M)] <- 0
log_msg("TWN: %d samples, %d TE nodes, %d IR nodes", nrow(M), ne, ni)

L <- matrix(0, n, n)
L[1:ne, 1:ne] <- num(opt$`lambda-tt`)
L[1:ne, (ne + 1):n] <- num(opt$`lambda-ti`); L[(ne + 1):n, 1:ne] <- num(opt$`lambda-ti`)
L[(ne + 1):n, (ne + 1):n] <- num(opt$`lambda-ii`)
diag(L) <- num(opt$`lambda-d`)
iso_gene <- iso2gene[isoforms]
gene_index <- setNames(seq_len(ne), genes)
for (g in unique(iso_gene)) {
    idx <- which(iso_gene == g)
    if (length(idx) >= 2) {
        pairs <- combn(idx, 2)
        for (k in seq_len(ncol(pairs))) {
            a <- ne + pairs[1, k]; b <- ne + pairs[2, k]
            L[a, b] <- num(opt$`lambda-s`); L[b, a] <- num(opt$`lambda-s`)
        }
    }
    if (!is.na(gene_index[g])) {
        gi <- gene_index[[g]]
        for (j in idx) { L[gi, ne + j] <- num(opt$`lambda-s`); L[ne + j, gi] <- num(opt$`lambda-s`) }
    }
}

S <- cov(M)
t0 <- Sys.time()
fit <- glasso::glasso(S, rho = L, maxit = as.integer(num(opt$`max-iter`)), penalize.diagonal = TRUE)
log_msg("glasso finished in %.1fs (%d iterations)", as.numeric(difftime(Sys.time(), t0, units = "secs")), fit$niter)
Theta <- fit$wi
nodes <- c(genes, isoforms)
node_gene <- c(genes, unname(iso_gene))
ut <- which(upper.tri(Theta) & Theta != 0, arr.ind = TRUE)
edges <- data.frame(node1 = nodes[ut[, 1]], node2 = nodes[ut[, 2]],
                    edge_type = ifelse(ut[, 2] <= ne, 1L, ifelse(ut[, 1] <= ne, 2L, 3L)),
                    weight = Theta[ut], stringsAsFactors = FALSE)
same_gene <- node_gene[ut[, 1]] == node_gene[ut[, 2]]
n_within <- sum(same_gene)
edges <- edges[!same_gene, ]
write.table(edges, file.path(opt$outdir, "edges.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
params <- sprintf('{"n_te": %d, "n_ir": %d, "n_edges": %d, "n_edges_tt": %d, "n_edges_ti": %d, "n_edges_ii": %d, "n_within_gene_removed": %d, "lambda_tt": %s, "lambda_ti": %s, "lambda_ii": %s, "lambda_d": %s, "lambda_s": %s, "glasso_iter": %d}',
                  ne, ni, nrow(edges), sum(edges$edge_type == 1), sum(edges$edge_type == 2), sum(edges$edge_type == 3), n_within,
                  num(opt$`lambda-tt`), num(opt$`lambda-ti`), num(opt$`lambda-ii`), num(opt$`lambda-d`), num(opt$`lambda-s`), fit$niter)
writeLines(params, file.path(opt$outdir, "params.json"))
log_msg("done: %d edges (TT %d, TI %d, II %d), %d within-gene edges removed", nrow(edges),
        sum(edges$edge_type == 1), sum(edges$edge_type == 2), sum(edges$edge_type == 3), n_within)
