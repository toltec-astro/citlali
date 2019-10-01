#pragma once

//----------------------------- o ---------------------------------------

/// calculates the 1d map psd
bool Map::calcMapPsd(double covCut) {
  // make sure we've got up to date coverage cut indices
  coverageCut = covCut;
  findWeightThresh();
  setCoverageCutRanges();

  // make sure our coverage cut map has an even number
  // of rows and columns
  int nx = cutXRange[1] - cutXRange[0] + 1;
  int ny = cutYRange[1] - cutYRange[0] + 1;
  int cxr0 = cutXRange[0];
  int cyr0 = cutYRange[0];
  int cxr1 = cutXRange[1];
  int cyr1 = cutYRange[1];
  if (nx % 2 == 1) {
    cxr1 = cutXRange[1] - 1;
    nx--;
  }
  if (ny % 2 == 1) {
    cyr1 = cutYRange[1] - 1;
    ny--;
  }

  // we will do the fft using fftw
  // here is the memory allocation and the plan setup
  fftw_complex *in;
  fftw_complex *out;
  fftw_plan *p;

#pragma omp critical(noiseFFT)
  {
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nx * ny);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * nx * ny);
    p = new fftw_plan;
    *p = fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    // cout << nx << " " << ny << " " << in << " " << out << " " << FFTW_FORWARD
    // << " " << FFTW_ESTIMATE << endl;
  }

  // the matrix to get fft'd is cast into vector form in *in;
  int ii, jj, stride, index;
  for (int i = cxr0; i <= cxr1; i++)
    for (int j = cyr0; j <= cyr1; j++) {
      ii = i - cxr0;
      jj = j - cyr0;
      stride = cyr1 - cyr0 + 1;
      index = stride * ii + jj;
      in[index][0] = image[i][j];
      in[index][1] = 0.;
    }

  // apply a hanning window
  MatDoub h = hanning(nx, ny);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      in[ny * i + j][0] *= h[i][j];

  // calculate frequencies
  double diffx = rowCoordsPhys[1] - rowCoordsPhys[0];
  double diffy = colCoordsPhys[1] - colCoordsPhys[0];
  double xsize = diffx * nx;
  double ysize = diffy * ny;
  double diffqx = 1. / xsize;
  double diffqy = 1. / ysize;

  // do the fft and cleanup the plan
  fftw_execute(*p);

#pragma omp critical(noiseFFT)
  {
    fftw_destroy_plan(*p);
    delete p;
  }
  // matching the idl code
  for (int i = 0; i < nx * ny; i++) {
    out[i][0] *= xsize * ysize / nx / ny;
    out[i][1] *= xsize * ysize / nx / ny;
  }

  // here is the magnitude
  // reuse h to be memory-kind, this is pmfq in the idl code
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      h[i][j] = diffqx * diffqy *
                (pow(out[ny * i + j][0], 2) + pow(out[ny * i + j][1], 2));

  VecDoub w(nx * ny);
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      w[ny * i + j] = h[i][j];

// free up resources
#pragma omp critical(noiseFFT)
  {
    fftw_free(in);
    fftw_free(out);
  }

  // vectors of frequencies
  VecDoub qx(nx);
  VecDoub qy(ny);
  int shift = nx / 2 - 1;
  for (int i = 0; i < nx; i++) {
    index = i - shift;
    if (index < 0)
      index += nx;
    qx[index] = diffqx * (i - (nx / 2 - 1));
  }
  shift = ny / 2 - 1;
  for (int i = 0; i < ny; i++) {
    index = i - shift;
    if (index < 0)
      index += ny;
    qy[index] = diffqy * (i - (ny / 2 - 1));
  }

  // shed first row and column of h, qx, qy
  MatDoub pmfq(nx - 1, ny - 1);
  for (int i = 1; i < nx; i++)
    for (int j = 1; j < ny; j++)
      pmfq[i - 1][j - 1] = h[i][j];
  for (int i = 0; i < nx - 1; i++)
    qx[i] = qx[i + 1];
  for (int j = 0; j < ny - 1; j++)
    qy[j] = qy[j + 1];

  // matrices of frequencies and distances
  MatDoub qmap(nx - 1, ny - 1);
  MatDoub qsymm(nx - 1, ny - 1);
  for (int i = 1; i < nx; i++)
    for (int j = 1; j < ny; j++) {
      qmap[i - 1][j - 1] = sqrt(pow(qx[i], 2) + pow(qy[j], 2));
      qsymm[i - 1][j - 1] = qx[i] * qy[j];
    }

  // find max of nx and ny and correspoinding diffq
  int nn;
  double diffq;
  if (nx > ny) {
    nn = nx / 2 + 1;
    diffq = diffqx;
  } else {
    nn = ny / 2 + 1;
    diffq = diffqy;
  }

  // generate the final vector of frequencies
  psdFreq.resize(nn);
  for (int i = 0; i < nn; i++)
    psdFreq[i] = diffq * (i + 0.5);

  // pack up the final vector of psd values
  psd.resize(nn);
  for (int i = 0; i < nn; i++) {
    int countS = 0;
    int countA = 0;
    double psdarrS = 0.;
    double psdarrA = 0.;
    for (int j = 0; j < nx - 1; j++)
      for (int k = 0; k < ny - 1; k++) {
        if ((int)(qmap[j][k] / diffq) == i && qsymm[j][k] >= 0.) {
          countS++;
          psdarrS += pmfq[j][k];
        }
        if ((int)(qmap[j][k] / diffq) == i && qsymm[j][k] < 0.) {
          countA++;
          psdarrA += pmfq[j][k];
        }
      }
    if (countS != 0)
      psdarrS /= countS;
    if (countA != 0)
      psdarrA /= countA;
    psd[i] = min(psdarrS, psdarrA);
  }

  // smooth the psd with a 10-element boxcar filter
  VecDoub tmp(nn);
  smooth_edge_truncate(psd, tmp, 10);
  for (int i = 0; i < nn; i++)
    psd[i] = tmp[i];

  psd2d = pmfq;
  psd2dFreq = qmap;

  return 1;
}
