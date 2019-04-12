using System;
using Xunit;
using OpenCvSharp;

namespace EllipseFitEvaluation.Tests
{
    public class TestEvaluateEllipseFit
    {
        [Fact]
        public void TestPerfectFit1()
        {
            string ellipseFilename = "2018-02-15 17.26.47.474000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(626.76f, 494.98f), new Size2f(387.96f, 381.45f), 170);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(1.0f, score);
        }

        [Fact]
        public void TestPerfectFit2()
        {
            string ellipseFilename = "2018-02-15 17.27.27.162000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(635.86f, 521.4f), new Size2f(168.05f, 165.09f), 164);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(1.0f, score);
        }

        [Fact]
        public void TestPerfectFit3()
        {
            string ellipseFilename = "2018-02-15 17.27.54.680000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(847.58f, 751.44f), new Size2f(33.93f, 30.67f), 18);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(1.0f, score);
        }

        [Fact]
        public void TestPartialFit1()
        {
            string ellipseFilename = "2018-02-15 17.26.47.474000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(630.0f, 500.0f), new Size2f(390.0f, 380.0f), 170);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(0.9909615f, score, 7);
        }

        [Fact]
        public void TestPartialFit2()
        {
            string ellipseFilename = "2018-02-15 17.27.27.162000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(600.0f, 500.0f), new Size2f(100.0f, 100.0f), 200);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(0.3624268f, score, 7);
        }

        [Fact]
        public void TestPartialFit3()
        {
            string ellipseFilename = "2018-02-15 17.27.54.680000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(900.0f, 800.0f), new Size2f(100.0f, 100.0f), 0);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(0.1081905f, score, 7);
        }

        [Fact]
        public void TestInvalidFit()
        {
            string ellipseFilename = "2018-02-15 17.26.47.474000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(200.0f, 100.0f), new Size2f(50.0f, 25.0f), 0);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(0.0, score);
        }
        
        [Fact]
        public void TestNonNullFitWithNullGt()
        {
            string ellipseFilename = "2018-02-15 17.36.17.793000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(200.0f, 100.0f), new Size2f(50.0f, 25.0f), 0);
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(0.0, score);
        }
        
        [Fact]
        public void TestNullFitWithNullGt()
        {
            string ellipseFilename = "2018-02-15 17.36.17.793000.tiff";
            Ellipse fitEllipse = null;
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(1.0, score);
        }
        
        [Fact]
        public void TestNullFitWithNonNullGt()
        {
            string ellipseFilename = "2018-02-15 17.26.47.474000.tiff";
            Ellipse fitEllipse = null;
            string csvFilename = "ground_truths_develop.csv";

            float score = EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename);

            Assert.Equal(0.0, score);
        }
        
        [Fact]
        public void TestInvalidTiffFilename()
        {
            string ellipseFilename = "nonexisting.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(200.0f, 100.0f), new Size2f(50.0f, 25.0f), 0);
            string csvFilename = "ground_truths_develop.csv";

            Assert.Throws<Exception>(() => EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename));
        }
        
        [Fact]
        public void TestInvalidCsvFilename()
        {
            string ellipseFilename = "2018-02-15 17.26.47.474000.tiff";
            Ellipse fitEllipse = new Ellipse(new Point2f(200.0f, 100.0f), new Size2f(50.0f, 25.0f), 0);
            string csvFilename = "nonexisting.csv";

            Assert.Throws<System.IO.FileNotFoundException>(() => EllipseFitEvaluation.EvaluateEllipseFit(ellipseFilename, fitEllipse, csvFilename));
        }
    }
}
