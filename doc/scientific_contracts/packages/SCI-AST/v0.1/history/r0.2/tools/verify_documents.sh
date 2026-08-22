#!/usr/bin/env bash
set -euo pipefail

# Keep the author-draft PDFs reproducible across verifier runs.
export SOURCE_DATE_EPOCH=1787270400

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
package_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_dir="$package_dir/src"
common_dir="$source_dir/common"
pdf_dir="$package_dir/pdf"
render_dir="$package_dir/rendered"
verification_dir="$package_dir/verification"
build_dir="$verification_dir/build"

required_commands=(pdfinfo pdftoppm shasum awk sed diff find wc grep)
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "missing required command: $command_name" >&2
    exit 1
  }
done

if command -v pdflatex >/dev/null 2>&1; then
  latex_engine=$(command -v pdflatex)
  latex_engine_kind=pdflatex
elif test -x /opt/homebrew/bin/tectonic; then
  latex_engine=/opt/homebrew/bin/tectonic
  latex_engine_kind=tectonic
elif command -v tectonic >/dev/null 2>&1; then
  latex_engine=$(command -v tectonic)
  latex_engine_kind=tectonic
else
  echo "missing required LaTeX engine: pdflatex or tectonic" >&2
  exit 1
fi

if test "$latex_engine_kind" = tectonic; then
  tectonic_cache_id=6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c
  tectonic_cache_root="$HOME/Library/Caches/Tectonic/bundles"
  tectonic_cache_data="$tectonic_cache_root/data/$tectonic_cache_id"
  tectonic_cache_hash="$tectonic_cache_root/hashes/https,58,,47,,47,relay.fullyjustified.net,47,default_bundle_v33.tar"
  tectonic_bundle=/private/tmp/sci-ast-v01-tectonic-bundle
  test -d "$tectonic_cache_data" || {
    echo "missing cached Tectonic bundle data: $tectonic_cache_data" >&2
    exit 1
  }
  test -s "$tectonic_cache_hash" || {
    echo "missing cached Tectonic bundle hash: $tectonic_cache_hash" >&2
    exit 1
  }
  mkdir -p "$tectonic_bundle"
  if ! test -s "$tectonic_bundle/SHA256SUM"; then
    cp -R "$tectonic_cache_data/." "$tectonic_bundle/"
    cp "$tectonic_cache_hash" "$tectonic_bundle/SHA256SUM"
  fi
fi

python_executable="$HOME/tolteca/bin/python"
if ! test -x "$python_executable"; then
  echo "missing required project Python: $python_executable" >&2
  exit 1
fi

required_sources=(
  "$common_dir/notation.tex"
  "$common_dir/definitions.tex"
  "$common_dir/equations.tex"
  "$common_dir/assumptions.tex"
  "$common_dir/requirements.tex"
  "$common_dir/edge_cases.tex"
  "$source_dir/scientific-rationale.tex"
  "$source_dir/engineering-conformance.tex"
)
for source_path in "${required_sources[@]}"; do
  test -s "$source_path" || {
    echo "missing or empty source: $source_path" >&2
    exit 1
  }
done

common_modules=(notation definitions equations assumptions requirements edge_cases)
for module in "${common_modules[@]}"; do
  engineering_count=$(grep -F -c "\\input{common/$module.tex}" \
    "$source_dir/engineering-conformance.tex")
  test "$engineering_count" -eq 1 || {
    echo "engineering-conformance must include common/$module.tex exactly once" >&2
    exit 1
  }
  rationale_count=$(grep -F -c "\\input{common/$module.tex}" \
    "$source_dir/scientific-rationale.tex" || true)
  test "$rationale_count" -eq 0 || {
    echo "scientific-rationale must remain a concise narrative, not include common/$module.tex" >&2
    exit 1
  }
done

figure_count=$(grep -F -c '\begin{figure}' "$source_dir/scientific-rationale.tex")
test "$figure_count" -eq 3 || {
  echo "scientific-rationale must contain exactly three explanatory figures" >&2
  exit 1
}

if grep -E 'Only nonpolarimetric|nonpolarimetric Stokes-I|nonpolarimetric Stokes I' \
  "$common_dir/assumptions.tex" "$common_dir/definitions.tex" \
  "$source_dir/scientific-rationale.tex"; then
  echo "ambiguous nonpolarimetric/Stokes-I wording remains" >&2
  exit 1
fi

ordinary_scope_count=$(grep -F -h 'Only the ordinary nonpolarimetric coordinate path' \
  "$common_dir/assumptions.tex" "$common_dir/definitions.tex" \
  "$source_dir/scientific-rationale.tex" | wc -l | awk '{print $1}')
test "$ordinary_scope_count" -eq 3 || {
  echo "ordinary nonpolarimetric coordinate path wording must appear in ASM-014, definitions, and narrative" >&2
  exit 1
}

for required_phrase in 'no demodulation' 'Stokes reconstruction' 'raw KID'; do
  grep -F "$required_phrase" "$common_dir/assumptions.tex" >/dev/null || {
    echo "ASM-014 missing required scope phrase: $required_phrase" >&2
    exit 1
  }
done

boundary_path="$package_dir/SCI-ALIGN_TO_SCI-AST_BOUNDARY.md"
boundary_sha=$(shasum -a 256 "$boundary_path" | awk '{print $1}')
test "$boundary_sha" = 359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf || {
  echo "unexpected final shared boundary digest: $boundary_sha" >&2
  exit 1
}

mkdir -p "$pdf_dir" "$render_dir" "$verification_dir" "$build_dir"

expected_req="$verification_dir/expected-requirements.txt"
actual_req="$verification_dir/actual-requirements.txt"
expected_pred="$verification_dir/expected-predictions.txt"
actual_pred="$verification_dir/actual-predictions.txt"
expected_asm="$verification_dir/expected-assumptions.txt"
actual_asm="$verification_dir/actual-assumptions.txt"

awk 'BEGIN { for (i=1; i<=90; i++) printf "%03d\n", i }' > "$expected_req"
sed -n 's/.*\\Req{\([0-9][0-9][0-9]\)}.*/\1/p' \
  "$common_dir/requirements.tex" > "$actual_req"
diff -u "$expected_req" "$actual_req"

awk 'BEGIN { for (i=1; i<=50; i++) printf "%03d\n", i }' > "$expected_pred"
sed -n 's/.*\\Pred{\([0-9][0-9][0-9]\)}.*/\1/p' \
  "$common_dir/edge_cases.tex" > "$actual_pred"
diff -u "$expected_pred" "$actual_pred"

awk 'BEGIN { for (i=1; i<=15; i++) printf "%03d\n", i }' > "$expected_asm"
sed -n 's/.*\\Assumption{\([0-9][0-9][0-9]\)}.*/\1/p' \
  "$common_dir/assumptions.tex" > "$actual_asm"
diff -u "$expected_asm" "$actual_asm"

cd "$source_dir"
documents=(scientific-rationale engineering-conformance)
for document in "${documents[@]}"; do
  if test "$latex_engine_kind" = pdflatex; then
    for pass in 1 2 3; do
      "$latex_engine" \
        -interaction=nonstopmode \
        -halt-on-error \
        -file-line-error \
        -output-directory="$build_dir" \
        "$document.tex" \
        > "$verification_dir/$document-pass-$pass.stdout"
    done
  else
    "$latex_engine" \
      --chatter minimal \
      --only-cached \
      --bundle "$tectonic_bundle" \
      --keep-intermediates \
      --keep-logs \
      --reruns 2 \
      --outdir "$build_dir" \
      "$document.tex" \
      > "$verification_dir/$document-tectonic.stdout" 2>&1
  fi

  built_pdf_path="$build_dir/$document.pdf"
  log_path="$build_dir/$document.log"
  pdf_path="$pdf_dir/$document.pdf"
  info_path="$verification_dir/$document-info.txt"
  structure_path="$verification_dir/$document-structure.json"
  document_render_dir="$render_dir/$document"

  test -s "$built_pdf_path"
  test -s "$log_path"

  if grep -E 'LaTeX Warning:|Reference .* undefined|Citation .* undefined|Overfull \\hbox|Overfull \\vbox' "$log_path"; then
    echo "fatal layout/reference warning in $log_path" >&2
    exit 1
  fi

  cp "$built_pdf_path" "$pdf_path"
  test -s "$pdf_path"

  pdfinfo "$pdf_path" > "$info_path"
  pages=$(awk '/^Pages:/ {print $2}' "$info_path")
  test -n "$pages"
  test "$pages" -gt 0
  if test "$document" = scientific-rationale; then
    test "$pages" -ge 8 -a "$pages" -le 10 || {
      echo "scientific-rationale must be 8-10 pages, got $pages" >&2
      exit 1
    }
  fi

  "$python_executable" "$script_dir/verify_pdf.py" "$pdf_path" > "$structure_path"
  test -s "$structure_path"

  mkdir -p "$document_render_dir"
  case "$document_render_dir" in
    "$package_dir"/rendered/*) find "$document_render_dir" -type f -name '*.png' -delete ;;
    *) echo "refusing unsafe render cleanup: $document_render_dir" >&2; exit 1 ;;
  esac
  pdftoppm -png -r 130 "$pdf_path" "$document_render_dir/page" \
    > "$verification_dir/$document-render.stdout" 2>&1
  rendered_pages=$(find "$document_render_dir" -type f -name 'page-*.png' | wc -l | awk '{print $1}')
  test "$rendered_pages" -eq "$pages"

  shasum -a 256 "$pdf_path"
  echo "$document: $pages pages rendered and structurally verified"
done

echo "SCI-AST document verification complete"
